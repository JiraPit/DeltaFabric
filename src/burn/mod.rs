pub mod traits;

#[cfg(test)]
mod test;

use crate::core::{
    access_archived_packet,
    config::Config,
    networking::Session,
    packet::{DeltaPacket, SparseDelta},
    sync::{apply_deltas, generate_local_delta, process_deltas},
};
use anyhow::{Context, Result};
use std::collections::HashMap;
use tracing::{info, warn};

pub use traits::{apply_params, extract_params};

/// Manages distributed model synchronization across nodes using delta compression.
///
/// Fabric coordinates parameter updates between multiple nodes by:
/// - Aggregating incoming deltas from peers
/// - Applying synced parameters to local models
/// - Broadcasting local deltas at configurable intervals
pub struct Fabric {
    /// Zenoh session for network communication
    pub session: Session,
    /// Configuration parameters for delta sync
    pub config: Config,
    /// Reference weights for delta computation (updated on each sync)
    pub anchor_weights: Vec<f32>,
    /// Tracks last seen sequence ID per origin node for deduplication
    pub seen_table: HashMap<u64, u64>,
    /// Local sequence counter for delta ordering
    pub local_sequence: u64,
    /// Total number of sync steps performed
    pub step_count: u64,
}

impl Fabric {
    /// Creates a new Fabric instance and initializes cluster discovery.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Unique identifier for this node (1-indexed)
    /// * `config` - Configuration containing peers and sync parameters
    ///
    /// # Errors
    ///
    /// Returns an error if Zenoh session creation or cluster initialization fails.
    pub async fn new(node_id: u64, config: Config) -> Result<Self> {
        info!(node_id = %node_id, "Initializing DeltaFabric");

        let node = crate::core::networking::Node {
            id: node_id,
            peers: config.peers.clone(),
        };

        let mut session = Session::new(node)
            .await
            .context("Failed to create session")?;

        session
            .init_fabric()
            .await
            .context("Failed to initialize fabric")?;

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        Ok(Self {
            session,
            config,
            anchor_weights: Vec::new(),
            seen_table: HashMap::new(),
            local_sequence: 0,
            step_count: 0,
        })
    }

    /// Performs one synchronization step, updating model with peer deltas.
    ///
    /// This method:
    /// 1. Extracts parameters from the model
    /// 2. Pulls and processes incoming delta packets from peers
    /// 3. Applies aggregated deltas to model weights
    /// 4. Generates and broadcasts local delta if sync interval reached
    ///
    /// # Arguments
    ///
    /// * `model` - The model to sync (taken by value, returned updated)
    ///
    /// # Type Parameters
    ///
    /// * `B` - Burn backend (e.g., NdArray)
    /// * `M` - Model type implementing Module
    ///
    /// # Returns
    ///
    /// Updated model with synced parameters applied.
    pub async fn step<B: burn::tensor::backend::Backend, M: burn::module::Module<B>>(
        &mut self,
        model: M,
    ) -> Result<M> {
        self.step_count += 1;
        let step_count = self.step_count;

        let mut active_flat = extract_params(&model);

        if self.anchor_weights.is_empty() {
            self.anchor_weights = active_flat.clone();
            info!(
                step = %step_count,
                num_weights = %active_flat.len(),
                "Initialized anchor weights"
            );
        }

        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut relay_updates: HashMap<u64, SparseDelta> = HashMap::new();

        for sample in self.session.pull_packets() {
            let payload = sample.payload().to_bytes();
            match access_archived_packet(&payload) {
                Ok(incoming) => {
                    if let Some(updates) = process_deltas(
                        &mut aggregator,
                        incoming,
                        &mut self.seen_table,
                        self.config.alpha,
                        self.config.relay_threshold,
                    ) {
                        relay_updates.extend(updates);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to deserialize incoming packet");
                }
            }
        }

        let num_peer_updates = aggregator.len();
        if num_peer_updates > 0 {
            info!(
                step = %step_count,
                num_updates = %num_peer_updates,
                "Applying peer deltas"
            );
        }

        apply_deltas(&mut active_flat, &mut self.anchor_weights, &aggregator);

        if step_count.is_multiple_of(self.config.sync_interval) {
            self.local_sequence += 1;
            if let Some(delta) = generate_local_delta(
                &active_flat,
                &mut self.anchor_weights,
                self.config.delta_selection_ratio,
                self.session.node.id,
                self.local_sequence,
            ) {
                info!(
                    step = %step_count,
                    seq = %self.local_sequence,
                    num_indices = %delta.indices.len(),
                    "Generated local delta"
                );
                relay_updates.insert(self.session.node.id, delta);
            }
        }

        if !relay_updates.is_empty() {
            let packet = DeltaPacket {
                updates: relay_updates,
            };
            self.session
                .broadcast(packet)
                .await
                .context("Failed to broadcast packet")?;
        }

        let model = apply_params(model, &active_flat);
        Ok(model)
    }

    /// Shuts down the Fabric, closing all network connections.
    ///
    /// Broadcasts an OFFLINE status to peers before closing.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!(node_id = %self.session.node.id, "Shutting down DeltaFabric");

        self.session
            .shutdown()
            .await
            .context("Failed to shutdown session")?;

        info!(node_id = %self.session.node.id, "DeltaFabric shutdown complete");
        Ok(())
    }
}
