use crate::core::codec::{ValueEncoding, decode_delta};
use crate::core::packet::{AnchorSnapshot, DeltaPacket};
use crate::core::sync::generate::generate_local_delta;
use crate::core::sync::mixing::{MixingWeights, accumulate, mix};
use anyhow::{Result, bail, ensure};
use std::collections::HashMap;
use tracing::{info, warn};

/// What happened to a delta packet passed to SyncState::receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiveOutcome {
    /// Added to the mixing sum
    Applied,
    /// Already applied earlier; ignored
    Duplicate,
    /// Held until the running resync completes
    Buffered,
    /// A missing packet or different initial weights were detected; a resync started
    ResyncStarted,
    /// Sender is not a neighbor; ignored
    UnknownPeer,
    /// Arrived before initialize(); ignored (a later gap triggers a resync)
    NotInitialized,
    /// Could not be decoded; ignored (the next packet's gap triggers a resync)
    Malformed,
}

/// Tracking state for one neighbor.
#[derive(Debug, Clone)]
struct PeerState {
    /// Mixing weight of this neighbor
    weight: f32,
    /// Sequence ID of the next delta expected from this neighbor
    next_seq: u64,
    /// Whether its initial weights still need to be checked against ours
    check_init: bool,
}

/// An in-progress rebuild of the mixing sum from neighbor snapshots.
#[derive(Debug, Default)]
struct Resync {
    /// Snapshots received so far, keyed by neighbor ID
    snapshots: HashMap<u64, AnchorSnapshot>,
    /// Deltas received while waiting, replayed once all snapshots arrive
    buffered: Vec<(u64, DeltaPacket)>,
}

/// Framework-independent state of the DeltaFabric sync protocol for one node.
///
/// Each node keeps three weight vectors: the model's active weights (owned by
/// the ML framework), its anchor (the copy of its weights its neighbors know
/// about) and the mixing sum (the weighted sum of its own and its neighbors'
/// anchors, rebuilt from received deltas). Only sparse anchor changes are
/// sent. At every sync the active weights move toward the mixing sum, which
/// pulls the nodes toward a common model (CHOCO-SGD).
#[derive(Debug)]
pub struct SyncState {
    gamma: f32,
    delta_selection_ratio: f32,
    value_encoding: ValueEncoding,
    self_weight: f32,
    peers: HashMap<u64, PeerState>,
    anchor: Vec<f32>,
    mix_sum: Vec<f32>,
    init_hash: u64,
    local_sequence: u64,
    resync: Option<Resync>,
}

impl SyncState {
    /// Creates an uninitialized SyncState.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Consensus step size
    /// * `delta_selection_ratio` - Fraction of parameters sent per sync
    /// * `weights` - Mixing weights for this node and its neighbors
    pub fn new(gamma: f32, delta_selection_ratio: f32, weights: MixingWeights) -> Self {
        let peers = weights
            .neighbors
            .iter()
            .map(|(&id, &weight)| {
                (
                    id,
                    PeerState {
                        weight,
                        next_seq: 1,
                        check_init: true,
                    },
                )
            })
            .collect();

        Self {
            gamma,
            delta_selection_ratio,
            value_encoding: ValueEncoding::default(),
            self_weight: weights.self_weight,
            peers,
            anchor: Vec::new(),
            mix_sum: Vec::new(),
            init_hash: 0,
            local_sequence: 0,
            resync: None,
        }
    }

    /// Sets how this node encodes the values it sends. Returns self for chaining.
    ///
    /// Receivers decode any encoding, so nodes may use different ones.
    pub fn with_encoding(mut self, value_encoding: ValueEncoding) -> Self {
        self.value_encoding = value_encoding;
        self
    }

    /// Whether initialize() has been called.
    pub fn is_initialized(&self) -> bool {
        !self.anchor.is_empty()
    }

    /// Sets the anchor and mixing sum to the model's initial weights.
    ///
    /// All nodes are expected to start from identical weights, so every
    /// neighbor's anchor (and therefore the weighted sum) equals our own.
    /// A node that started differently is caught by its init hash.
    ///
    /// # Arguments
    ///
    /// * `active` - The model's initial weights
    pub fn initialize(&mut self, active: &[f32]) {
        self.anchor = active.to_vec();
        self.mix_sum = active.to_vec();
        self.init_hash = weights_hash(active);
        info!(num_weights = %active.len(), "Initialized anchor weights");
    }

    /// This node's anchor.
    pub fn anchor(&self) -> &[f32] {
        &self.anchor
    }

    /// Weighted sum of own and neighbor anchors.
    pub fn mix_sum(&self) -> &[f32] {
        &self.mix_sum
    }

    /// Sequence ID of the last delta this node sent.
    pub fn local_sequence(&self) -> u64 {
        self.local_sequence
    }

    /// Whether a resync is in progress (mixing is paused).
    pub fn is_resyncing(&self) -> bool {
        self.resync.is_some()
    }

    /// Handles a delta packet from a neighbor.
    ///
    /// Deltas must arrive exactly once and in order. A duplicate is ignored;
    /// a gap in sequence IDs (or a mismatching init hash) means the mixing
    /// sum can no longer be trusted, so a resync starts.
    ///
    /// # Arguments
    ///
    /// * `origin` - ID of the sending node
    /// * `packet` - The received packet
    ///
    /// # Returns
    ///
    /// What was done with the packet.
    pub fn receive(&mut self, origin: u64, packet: DeltaPacket) -> ReceiveOutcome {
        if !self.is_initialized() {
            return ReceiveOutcome::NotInitialized;
        }
        let Some(peer) = self.peers.get(&origin) else {
            warn!(origin = %origin, "Ignoring delta from a node that is not a neighbor");
            return ReceiveOutcome::UnknownPeer;
        };
        if let Some(resync) = &mut self.resync {
            resync.buffered.push((origin, packet));
            return ReceiveOutcome::Buffered;
        }

        let seq = packet.delta.sequence_id;
        if seq < peer.next_seq {
            return ReceiveOutcome::Duplicate;
        }
        if seq > peer.next_seq {
            warn!(origin = %origin, expected = %peer.next_seq, got = %seq, "Missing delta; starting resync");
            return self.start_resync(origin, packet);
        }
        if peer.check_init
            && let Some(hash) = packet.init_hash
            && hash != self.init_hash
        {
            warn!(origin = %origin, "Peer started from different weights; starting resync");
            return self.start_resync(origin, packet);
        }

        let delta = match decode_delta(&packet.delta) {
            Ok(delta) => delta,
            Err(e) => {
                warn!(origin = %origin, seq = %seq, error = %e, "Ignoring malformed delta");
                return ReceiveOutcome::Malformed;
            }
        };

        let peer = self.peers.get_mut(&origin).expect("peer checked above");
        accumulate(&mut self.mix_sum, &delta, peer.weight);
        peer.next_seq += 1;
        peer.check_init = false;
        ReceiveOutcome::Applied
    }

    /// Runs one sync: mixes toward neighbors and sends the top-k anchor change.
    ///
    /// # Arguments
    ///
    /// * `active` - Current model weights (mutated in place by the mixing step)
    ///
    /// # Returns
    ///
    /// The packet to broadcast, or None if there was nothing to send.
    ///
    /// # Errors
    ///
    /// Returns an error if not initialized or if the weight count changed.
    pub fn sync_step(&mut self, active: &mut [f32]) -> Result<Option<DeltaPacket>> {
        ensure!(self.is_initialized(), "sync_step called before initialize");
        ensure!(
            active.len() == self.anchor.len(),
            "Model has {} weights but the anchor has {}",
            active.len(),
            self.anchor.len()
        );

        // Mix before sending, so our own anchor and the neighbor anchors in the
        // mixing sum describe the same exchange; the pull is sent with the next
        // delta. A mixing sum missing deltas would pull toward a wrong average,
        // so mixing pauses during a resync.
        if self.resync.is_none() && self.gamma > 0.0 {
            mix(active, &self.anchor, &self.mix_sum, self.gamma);
        }

        let packet = generate_local_delta(
            active,
            &mut self.anchor,
            self.delta_selection_ratio,
            self.local_sequence + 1,
            self.value_encoding,
        )
        .map(|(sent, encoded)| {
            self.local_sequence += 1;
            accumulate(&mut self.mix_sum, &sent, self.self_weight);
            DeltaPacket {
                delta: encoded,
                init_hash: (self.local_sequence == 1).then_some(self.init_hash),
            }
        });

        Ok(packet)
    }

    /// A full copy of this node's anchor, for a neighbor that is resyncing.
    pub fn snapshot(&self) -> AnchorSnapshot {
        AnchorSnapshot {
            sequence_id: self.local_sequence,
            anchor: self.anchor.clone(),
        }
    }

    /// Neighbors whose snapshot is still needed by the running resync.
    pub fn missing_snapshots(&self) -> Vec<u64> {
        match &self.resync {
            Some(resync) => {
                let mut missing: Vec<u64> = self
                    .peers
                    .keys()
                    .filter(|id| !resync.snapshots.contains_key(id))
                    .copied()
                    .collect();
                missing.sort_unstable();
                missing
            }
            None => Vec::new(),
        }
    }

    /// Handles a neighbor's anchor snapshot during a resync.
    ///
    /// Once every neighbor's snapshot has arrived, the mixing sum is rebuilt
    /// from them and the deltas buffered meanwhile are replayed.
    ///
    /// # Arguments
    ///
    /// * `origin` - ID of the neighbor that sent the snapshot
    /// * `snapshot` - Its anchor and last sequence ID
    ///
    /// # Returns
    ///
    /// True if this snapshot completed the resync.
    ///
    /// # Errors
    ///
    /// Returns an error if no resync is running, the sender is not a
    /// neighbor, or the snapshot has the wrong number of weights.
    pub fn receive_snapshot(&mut self, origin: u64, snapshot: AnchorSnapshot) -> Result<bool> {
        if !self.peers.contains_key(&origin) {
            bail!("Snapshot from {origin}, which is not a neighbor");
        }
        ensure!(
            snapshot.anchor.len() == self.anchor.len(),
            "Snapshot from {origin} has {} weights, expected {}",
            snapshot.anchor.len(),
            self.anchor.len()
        );
        let Some(resync) = &mut self.resync else {
            bail!("Unexpected snapshot from {origin}: no resync is running");
        };

        resync.snapshots.insert(origin, snapshot);
        if resync.snapshots.len() < self.peers.len() {
            return Ok(false);
        }

        let resync = self.resync.take().expect("resync checked above");
        self.mix_sum = self.anchor.iter().map(|a| self.self_weight * a).collect();
        for (id, snapshot) in &resync.snapshots {
            let peer = self.peers.get_mut(id).expect("snapshots only from peers");
            for (sum, &value) in self.mix_sum.iter_mut().zip(&snapshot.anchor) {
                *sum += peer.weight * value;
            }
            peer.next_seq = snapshot.sequence_id + 1;
            peer.check_init = false;
        }
        info!(peers = %resync.snapshots.len(), "Resync complete");

        let mut buffered = resync.buffered;
        buffered.sort_by_key(|(_, packet)| packet.delta.sequence_id);
        for (id, packet) in buffered {
            self.receive(id, packet);
        }
        Ok(true)
    }

    /// Starts a resync and buffers the packet that triggered it.
    fn start_resync(&mut self, origin: u64, packet: DeltaPacket) -> ReceiveOutcome {
        self.resync = Some(Resync {
            snapshots: HashMap::new(),
            buffered: vec![(origin, packet)],
        });
        ReceiveOutcome::ResyncStarted
    }
}

/// Hashes weights with FNV-1a over their bit patterns.
///
/// Stable across processes and builds, so nodes can compare initial weights.
pub fn weights_hash(weights: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for w in weights {
        for byte in w.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}
