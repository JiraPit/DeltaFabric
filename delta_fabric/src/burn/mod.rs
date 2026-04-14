pub mod flatten;
pub mod unflatten;

#[cfg(test)]
mod test;

use crate::core::{
    config::FabricConfig,
    networking::Session,
    packet::{FabricPacket, SparseDelta},
    sync::{apply_deltas, generate_local_delta, process_deltas},
};
use anyhow::{Context, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use std::collections::HashMap;
use std::marker::PhantomData;
use tracing::{info, warn};

pub use flatten::flatten_burn_model;
pub use unflatten::unflatten_burn_model;

pub struct Fabric<B: Backend> {
    pub session: Session,
    pub config: FabricConfig,
    pub anchor_weights: Vec<f32>,
    pub seen_table: HashMap<u64, u64>,
    pub local_sequence: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> Fabric<B> {
    pub async fn new(node_id: u64, config: FabricConfig) -> Result<Self> {
        info!(node_id = %node_id, "Initializing DeltaFabric");

        let node = crate::core::networking::Node {
            id: node_id,
            expected_peers: config.expected_peers.clone(),
        };

        let mut session = Session::new(node)
            .await
            .context("Failed to create session")?;

        session
            .init_cluster()
            .await
            .context("Failed to initialize cluster")?;

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        Ok(Self {
            session,
            config,
            anchor_weights: Vec::new(),
            seen_table: HashMap::new(),
            local_sequence: 0,
            _backend: PhantomData,
        })
    }

    pub async fn step<M: Module<B>>(&mut self, model: M, step_count: u64) -> Result<M> {
        let mut active_flat = flatten_burn_model(&model).context("Failed to flatten model")?;

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
            match rkyv::from_bytes::<FabricPacket, rkyv::rancor::Error>(&payload) {
                Ok(incoming) => {
                    if let Some(updates) = process_deltas(
                        &mut aggregator,
                        &incoming,
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
                self.config.top_k_pct,
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
            let packet = FabricPacket {
                updates: relay_updates,
            };
            self.session
                .broadcast(packet)
                .await
                .context("Failed to broadcast packet")?;
        }

        let model =
            unflatten_burn_model(model, &active_flat).context("Failed to unflatten model")?;

        Ok(model)
    }

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
