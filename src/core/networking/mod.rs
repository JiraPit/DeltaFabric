use crate::core::packet::{AnchorSnapshot, DeltaPacket, to_bytes};
use crate::core::sync::{MixingWeights, metropolis_weights};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::vec::Vec;
use tracing::{error, info, warn};
use zenoh::Session as ZenohSession;
use zenoh::handlers::FifoChannelHandler;
use zenoh::qos::CongestionControl;
use zenoh::query::{Query, Reply};

/// How long a snapshot request waits for a reply.
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(60);

/// Minimum time between two snapshot requests to the same peer.
const SNAPSHOT_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// Represents the state of a node in the cluster.
///
/// Published to peers for cluster discovery and health monitoring.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct NodeState {
    /// IDs of peers this node is connected to
    pub peers: Vec<u64>,
    /// Current status: "DISCOVERING", "READY", or "OFFLINE"
    pub status: String,
    /// Neighbors this node will exchange deltas with (set once READY)
    #[serde(default)]
    pub neighbors: Vec<u64>,
}

/// Represents a node in the distributed cluster.
#[derive(Clone, Debug)]
pub struct Node {
    /// Unique identifier for this node within its cluster
    pub id: u64,
    /// List of peer node IDs
    pub peers: Vec<u64>,
    /// Name of the training job; nodes only see nodes with the same cluster
    pub cluster: String,
}

/// Snapshot of the traffic counters of a Session.
///
/// Byte counts are serialized payload sizes (excluding Zenoh framing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrafficStats {
    /// Number of delta packets broadcast
    pub packets_sent: u64,
    /// Total payload bytes broadcast
    pub bytes_sent: u64,
    /// Number of delta packets received from peers
    pub packets_received: u64,
    /// Total payload bytes received from peers
    pub bytes_received: u64,
    /// Number of anchor snapshots sent to resyncing peers
    pub snapshots_sent: u64,
    /// Total payload bytes of snapshots sent
    pub snapshot_bytes_sent: u64,
    /// Number of anchor snapshots received during resyncs
    pub snapshots_received: u64,
    /// Total payload bytes of snapshots received
    pub snapshot_bytes_received: u64,
}

/// Thread-safe traffic counters backing TrafficStats.
#[derive(Default)]
struct TrafficCounters {
    packets_sent: AtomicU64,
    bytes_sent: AtomicU64,
    packets_received: AtomicU64,
    bytes_received: AtomicU64,
    snapshots_sent: AtomicU64,
    snapshot_bytes_sent: AtomicU64,
    snapshots_received: AtomicU64,
    snapshot_bytes_received: AtomicU64,
}

/// A snapshot request waiting for its reply.
struct PendingSnapshot {
    /// Channel receiving the reply (closed when the query finishes)
    replies: Option<FifoChannelHandler<Reply>>,
    /// When the request was sent
    sent_at: Instant,
}

/// Builds the key expression "deltafabric/{cluster}/node/{node}/{kind}".
///
/// `node` may be "*" to match every node of the cluster.
fn cluster_key(cluster: &str, node: impl std::fmt::Display, kind: &str) -> String {
    format!("deltafabric/{cluster}/node/{node}/{kind}")
}

/// Extracts the node ID from a key expression ending in ".../{id}/{kind}".
fn node_id_from_key(key: &str) -> Option<u64> {
    key.rsplit('/').nth(1)?.parse().ok()
}

/// Checks that every neighbor also counts this node as its neighbor.
///
/// Mixing weights are only symmetric if both ends of every link agree. A node
/// that finished discovery before hearing from a peer that lists it ends up
/// without that link, which this detects.
///
/// # Arguments
///
/// * `my_id` - This node's ID
/// * `my_neighbors` - This node's neighbors
/// * `known_states` - Latest READY states of the cluster nodes
///
/// # Errors
///
/// Returns an error naming a neighbor that does not list this node.
pub(crate) fn check_neighbor_agreement(
    my_id: u64,
    my_neighbors: &[u64],
    known_states: &HashMap<u64, NodeState>,
) -> Result<()> {
    for peer in my_neighbors {
        let agrees = known_states
            .get(peer)
            .is_some_and(|state| state.neighbors.contains(&my_id));
        if !agrees {
            anyhow::bail!(
                "Node {peer} finished discovery without node {my_id} as a neighbor. \
                 List the link in both nodes' peers, or start {peer} after {my_id}."
            );
        }
    }
    Ok(())
}

/// Manages Zenoh network communication for delta synchronization.
///
/// Handles cluster discovery, peer communication, delta broadcasting and
/// anchor snapshots for resyncing peers.
pub struct Session {
    zenoh: ZenohSession,
    /// Local node information
    pub node: Node,
    /// Undirected cluster graph learned during discovery
    edges: HashMap<u64, HashSet<u64>>,
    /// Buffer for received delta samples
    delta_samples: Arc<Mutex<Vec<zenoh::sample::Sample>>>,
    /// Publisher for broadcasting local deltas
    delta_publisher: Option<zenoh::pubsub::Publisher<'static>>,
    /// Subscribers for receiving deltas from peers (must be kept alive)
    delta_subscribers: Vec<zenoh::pubsub::Subscriber<()>>,
    /// Buffer for snapshot requests from resyncing peers
    snapshot_queries: Arc<Mutex<Vec<Query>>>,
    /// Queryable answering snapshot requests (must be kept alive)
    snapshot_queryable: Option<zenoh::query::Queryable<()>>,
    /// Outstanding snapshot requests, keyed by peer ID
    pending_snapshots: HashMap<u64, PendingSnapshot>,
    /// Counters for delta traffic
    traffic: TrafficCounters,
}

impl Session {
    /// Creates a new Session with the given node configuration.
    ///
    /// # Arguments
    ///
    /// * `node` - Node struct with id and peer list
    ///
    /// # Errors
    ///
    /// Returns an error if Zenoh session creation fails.
    pub async fn new(node: Node) -> Result<Self> {
        let zenoh = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open Zenoh session: {}", e))?;

        info!(node_id = %node.id, "Opened Zenoh session");

        Ok(Self {
            zenoh,
            node,
            edges: HashMap::new(),
            delta_samples: Arc::new(Mutex::new(Vec::new())),
            delta_publisher: None,
            delta_subscribers: Vec::new(),
            snapshot_queries: Arc::new(Mutex::new(Vec::new())),
            snapshot_queryable: None,
            pending_snapshots: HashMap::new(),
            traffic: TrafficCounters::default(),
        })
    }

    /// Initializes the fabric by performing cluster discovery.
    ///
    /// Waits for all peers to become READY before completing.
    /// Sets up delta publishers, subscribers and the snapshot queryable after discovery.
    pub async fn init_fabric(&mut self) -> Result<()> {
        let my_topic = self.key(self.node.id, "state");
        let publisher = self
            .zenoh
            .declare_publisher(&my_topic)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to declare state publisher: {}", e))?;

        let state_sub = self
            .zenoh
            .declare_subscriber(cluster_key(&self.node.cluster, "*", "state"))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to declare state subscriber: {}", e))?;

        info!(node_id = %self.node.id, cluster = %self.node.cluster, peers = ?self.node.peers, "Starting cluster initialization");

        let mut known_states: HashMap<u64, NodeState> = HashMap::new();
        let mut global_nodes: HashSet<u64> = HashSet::new();
        let mut is_ready = false;

        loop {
            let my_state = NodeState {
                peers: self.node.peers.clone(),
                status: if is_ready { "READY" } else { "DISCOVERING" }.to_string(),
                neighbors: if is_ready { self.neighbors() } else { vec![] },
            };

            let state_json =
                serde_json::to_string(&my_state).context("Failed to serialize NodeState")?;

            publisher
                .put(state_json)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to publish state: {}", e))?;

            let mut samples_to_process = Vec::new();
            loop {
                let read_timeout =
                    tokio::time::timeout(Duration::from_millis(50), state_sub.recv_async());
                match read_timeout.await {
                    Ok(Ok(sample)) => {
                        samples_to_process.push(sample);
                        if samples_to_process.len() >= 100 {
                            break;
                        }
                    }
                    _ => break,
                }
            }

            for sample in samples_to_process {
                if let Some(incoming_id) = node_id_from_key(sample.key_expr().as_str()) {
                    match serde_json::from_slice::<NodeState>(&sample.payload().to_bytes()) {
                        Ok(state) => {
                            known_states.insert(incoming_id, state);
                        }
                        Err(e) => {
                            warn!(node_id = %incoming_id, error = %e, "Failed to parse NodeState from peer");
                        }
                    }
                }
            }
            known_states.insert(self.node.id, my_state.clone());

            let mut edges: HashMap<u64, HashSet<u64>> = HashMap::new();
            for (&id, state) in &known_states {
                if state.status == "OFFLINE" {
                    continue;
                }
                for &peer in &state.peers {
                    edges.entry(id).or_default().insert(peer);
                    edges.entry(peer).or_default().insert(id);
                }
            }

            let (mut visited, mut queue, mut missing) = (HashSet::new(), VecDeque::new(), false);
            queue.push_back(self.node.id);

            while let Some(curr) = queue.pop_front() {
                if !visited.insert(curr) {
                    continue;
                }
                match known_states.get(&curr) {
                    Some(s) if s.status != "OFFLINE" => {
                        if let Some(neighbors) = edges.get(&curr) {
                            for &n in neighbors {
                                queue.push_back(n);
                            }
                        }
                    }
                    _ => {
                        missing = true;
                        break;
                    }
                }
            }

            if !missing && !is_ready {
                global_nodes = visited.clone();
                // Keep only the connected component this node belongs to.
                edges.retain(|id, _| global_nodes.contains(id));
                self.edges = edges;
                // Declared before announcing READY: a peer that sees our READY
                // has also received our subscriptions, so no early delta is lost.
                self.declare_delta_endpoints().await?;
                is_ready = true;
                info!(node_count = %global_nodes.len(), "Discovery complete, waiting for READY signals");
            }

            if is_ready
                && global_nodes
                    .iter()
                    .all(|id| known_states.get(id).is_some_and(|s| s.status == "READY"))
            {
                break;
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        check_neighbor_agreement(self.node.id, &self.neighbors(), &known_states)?;

        info!(node_id = %self.node.id, neighbors = ?self.neighbors(), "Cluster initialization complete");

        Ok(())
    }

    /// Declares the delta publisher, snapshot queryable and neighbor subscribers.
    async fn declare_delta_endpoints(&mut self) -> Result<()> {
        // Block instead of dropping under congestion: a dropped delta forces a full resync.
        self.delta_publisher = Some(
            self.zenoh
                .declare_publisher(self.key(self.node.id, "delta"))
                .congestion_control(CongestionControl::Block)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to declare delta publisher: {}", e))?,
        );

        let queries = self.snapshot_queries.clone();
        self.snapshot_queryable = Some(
            self.zenoh
                .declare_queryable(self.key(self.node.id, "anchor"))
                .callback(move |query| {
                    if let Ok(mut guard) = queries.lock() {
                        guard.push(query);
                    }
                })
                .await
                .map_err(|e| anyhow::anyhow!("Failed to declare snapshot queryable: {}", e))?,
        );

        let samples_ref = self.delta_samples.clone();
        for peer_id in self.neighbors() {
            let samples = samples_ref.clone();
            let sub = self
                .zenoh
                .declare_subscriber(self.key(peer_id, "delta"))
                .callback(move |sample| {
                    if let Ok(mut guard) = samples.lock() {
                        guard.push(sample);
                    }
                })
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to declare delta subscriber for peer {}: {}",
                        peer_id,
                        e
                    )
                })?;
            self.delta_subscribers.push(sub);
        }

        Ok(())
    }

    /// Key expression of one of this cluster's topics for `node`.
    fn key(&self, node: u64, kind: &str) -> String {
        cluster_key(&self.node.cluster, node, kind)
    }

    /// Sorted IDs of this node's neighbors in the undirected cluster graph.
    ///
    /// A node is a neighbor if either side lists the other as a peer.
    /// Only meaningful after init_fabric.
    pub fn neighbors(&self) -> Vec<u64> {
        let mut neighbors: Vec<u64> = self
            .edges
            .get(&self.node.id)
            .map(|n| n.iter().copied().filter(|&p| p != self.node.id).collect())
            .unwrap_or_default();
        neighbors.sort_unstable();
        neighbors
    }

    /// Metropolis mixing weights for this node, from the discovered graph.
    ///
    /// Only meaningful after init_fabric.
    pub fn mixing_weights(&self) -> MixingWeights {
        metropolis_weights(self.node.id, &self.edges)
    }

    /// Retrieves and clears all received delta packets.
    ///
    /// # Returns
    ///
    /// Vector of (origin node ID, payload bytes) pairs.
    pub fn pull_packets(&self) -> Vec<(u64, Vec<u8>)> {
        let mut samples = Vec::new();
        if let Ok(mut guard) = self.delta_samples.lock() {
            samples.extend(guard.drain(..));
        }

        let mut packets = Vec::with_capacity(samples.len());
        for sample in samples {
            let payload = sample.payload().to_bytes().into_owned();
            self.traffic
                .packets_received
                .fetch_add(1, Ordering::Relaxed);
            self.traffic
                .bytes_received
                .fetch_add(payload.len() as u64, Ordering::Relaxed);
            match node_id_from_key(sample.key_expr().as_str()) {
                Some(origin) => packets.push((origin, payload)),
                None => warn!(key = %sample.key_expr(), "Delta on unexpected key"),
            }
        }
        packets
    }

    /// Broadcasts a delta packet to all peers.
    ///
    /// # Arguments
    ///
    /// * `packet` - The DeltaPacket to broadcast
    pub async fn broadcast(&self, packet: DeltaPacket) -> Result<()> {
        let bytes = to_bytes(&packet).context("Serialization of DeltaPacket failed")?;

        if let Some(publ) = &self.delta_publisher {
            let len = bytes.len() as u64;
            publ.put(bytes)
                .await
                .map_err(|e| anyhow::anyhow!("Zenoh broadcast failed: {}", e))?;
            self.traffic.packets_sent.fetch_add(1, Ordering::Relaxed);
            self.traffic.bytes_sent.fetch_add(len, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Retrieves and clears snapshot requests from resyncing peers.
    ///
    /// Each query must be answered with reply_snapshot. Dropping a query
    /// without replying makes the requester retry later.
    pub fn pull_snapshot_requests(&self) -> Vec<Query> {
        match self.snapshot_queries.lock() {
            Ok(mut guard) => guard.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Answers a snapshot request with this node's anchor.
    ///
    /// # Arguments
    ///
    /// * `query` - A request returned by pull_snapshot_requests
    /// * `snapshot` - This node's current anchor snapshot
    pub async fn reply_snapshot(&self, query: Query, snapshot: &AnchorSnapshot) -> Result<()> {
        let bytes = to_bytes(snapshot).context("Serialization of AnchorSnapshot failed")?;
        let len = bytes.len() as u64;
        query
            .reply(query.key_expr().clone(), bytes)
            .await
            .map_err(|e| anyhow::anyhow!("Snapshot reply failed: {}", e))?;
        self.traffic.snapshots_sent.fetch_add(1, Ordering::Relaxed);
        self.traffic
            .snapshot_bytes_sent
            .fetch_add(len, Ordering::Relaxed);
        Ok(())
    }

    /// Asks a peer for its anchor snapshot, unless a request is already pending.
    ///
    /// A failed request is retried at most once per SNAPSHOT_RETRY_INTERVAL.
    ///
    /// # Arguments
    ///
    /// * `peer` - ID of the peer to ask
    pub async fn request_snapshot(&mut self, peer: u64) -> Result<()> {
        if let Some(pending) = self.pending_snapshots.get(&peer)
            && (pending.replies.is_some() || pending.sent_at.elapsed() < SNAPSHOT_RETRY_INTERVAL)
        {
            return Ok(());
        }

        let replies = self
            .zenoh
            .get(self.key(peer, "anchor"))
            .timeout(SNAPSHOT_TIMEOUT)
            .await
            .map_err(|e| anyhow::anyhow!("Snapshot request to {} failed: {}", peer, e))?;
        info!(peer = %peer, "Requested anchor snapshot");
        self.pending_snapshots.insert(
            peer,
            PendingSnapshot {
                replies: Some(replies),
                sent_at: Instant::now(),
            },
        );
        Ok(())
    }

    /// Collects snapshot replies that have arrived.
    ///
    /// # Returns
    ///
    /// Vector of (peer ID, snapshot bytes) pairs.
    pub fn poll_snapshots(&mut self) -> Vec<(u64, Vec<u8>)> {
        let mut received = Vec::new();
        for (&peer, pending) in self.pending_snapshots.iter_mut() {
            let Some(replies) = &pending.replies else {
                continue;
            };
            match replies.try_recv() {
                Ok(Some(reply)) => {
                    match reply.result() {
                        Ok(sample) => {
                            let bytes = sample.payload().to_bytes().into_owned();
                            self.traffic
                                .snapshots_received
                                .fetch_add(1, Ordering::Relaxed);
                            self.traffic
                                .snapshot_bytes_received
                                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                            received.push((peer, bytes));
                        }
                        Err(e) => warn!(peer = %peer, error = ?e, "Snapshot request failed"),
                    }
                    pending.replies = None;
                }
                Ok(None) => {}
                // Query finished without a reply (peer not ready or timed out): retry later.
                Err(_) => pending.replies = None,
            }
        }
        received
    }

    /// Returns a snapshot of the traffic counters.
    pub fn stats(&self) -> TrafficStats {
        TrafficStats {
            packets_sent: self.traffic.packets_sent.load(Ordering::Relaxed),
            bytes_sent: self.traffic.bytes_sent.load(Ordering::Relaxed),
            packets_received: self.traffic.packets_received.load(Ordering::Relaxed),
            bytes_received: self.traffic.bytes_received.load(Ordering::Relaxed),
            snapshots_sent: self.traffic.snapshots_sent.load(Ordering::Relaxed),
            snapshot_bytes_sent: self.traffic.snapshot_bytes_sent.load(Ordering::Relaxed),
            snapshots_received: self.traffic.snapshots_received.load(Ordering::Relaxed),
            snapshot_bytes_received: self.traffic.snapshot_bytes_received.load(Ordering::Relaxed),
        }
    }

    /// Shuts down the session, broadcasting OFFLINE status to peers.
    pub async fn shutdown(&mut self) -> Result<()> {
        let state = NodeState {
            peers: self.node.peers.clone(),
            status: "OFFLINE".to_string(),
            neighbors: vec![],
        };

        let state_json =
            serde_json::to_string(&state).context("Failed to serialize OFFLINE state")?;

        match self
            .zenoh
            .put(self.key(self.node.id, "state"), state_json)
            .await
        {
            Ok(_) => {
                info!(node_id = %self.node.id, "Broadcast OFFLINE status");
            }
            Err(e) => {
                error!(node_id = %self.node.id, error = %e, "Failed to broadcast OFFLINE status");
            }
        }

        self.delta_publisher = None;
        self.snapshot_queryable = None;
        Ok(())
    }
}

#[cfg(test)]
mod test;
