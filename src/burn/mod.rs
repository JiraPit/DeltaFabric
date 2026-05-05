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
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::collections::HashMap;
use tracing::{info, warn};

/// Extracts all learnable parameters and buffers from a Burn model into a flat vector.
///
/// Uses visitation order to ensure deterministic flattening across nodes.
///
/// # Arguments
///
/// * `model` - The Burn model to extract parameters from
///
/// # Returns
///
/// Flat vector of all parameter values in deterministic order.
pub(crate) fn extract_params<M: Module<B>, B: Backend>(model: &M) -> Vec<f32> {
    let mut collector = ParamCollector { data: Vec::new() };
    model.visit(&mut collector);
    collector.data
}

/// Collects parameter values from a model during visitation.
struct ParamCollector {
    data: Vec<f32>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let values: Vec<f32> = param.val().to_data().into_vec::<f32>().unwrap();
        self.data.extend(values);
    }
}

/// Applies a flat parameter vector to a Burn model, returning the updated model.
///
/// Uses the same visitation order to ensure deterministic unflattening.
///
/// # Arguments
///
/// * `model` - The Burn model to update
/// * `params` - Flat vector of parameter values
///
/// # Returns
///
/// The updated model with new parameter values applied.
pub(crate) fn apply_params<M: Module<B>, B: Backend>(model: M, params: &[f32]) -> M {
    let mut setter = ParamSetter {
        data: params.to_vec(),
        pos: 0,
    };
    model.map(&mut setter)
}

/// Applies parameter updates from a flat vector to a model.
struct ParamSetter {
    data: Vec<f32>,
    pos: usize,
}

impl<B: Backend> ModuleMapper<B> for ParamSetter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let shape = param.val().dims();
        let num_elements = shape.iter().product::<usize>();
        let end_pos = (self.pos + num_elements).min(self.data.len());
        let tensor_data = self.data[self.pos..end_pos].to_vec();
        self.pos = end_pos;

        let device = param.val().device();
        let data = TensorData::new(tensor_data, shape);
        let requires_grad = param.val().is_require_grad();

        param.map(|_| {
            let tensor = Tensor::from_data(data, &device);
            if requires_grad {
                tensor.require_grad()
            } else {
                tensor
            }
        })
    }
}

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
    pub async fn step<B: Backend, M: Module<B>>(&mut self, model: M) -> Result<M> {
        self.step_count += 1;
        let step_count = self.step_count;
        let my_id = self.session.node.id;
        let sync_interval = self.config.sync_interval;

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
                        my_id,
                    ) {
                        relay_updates.extend(updates);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to deserialize incoming packet");
                }
            }
        }

        let is_sync_step = step_count.is_multiple_of(sync_interval);
        let has_peer_updates = !aggregator.is_empty();
        let need_active = self.anchor_weights.is_empty() || has_peer_updates || is_sync_step;

        if !need_active {
            return Ok(model);
        }

        let mut active_flat = extract_params(&model);

        if self.anchor_weights.is_empty() {
            self.anchor_weights = active_flat.clone();
            info!(
                step = %step_count,
                num_weights = %active_flat.len(),
                "Initialized anchor weights"
            );
        }

        if has_peer_updates {
            info!(node_id = %my_id, count = %aggregator.len(), "Applying peer deltas");
            apply_deltas(&mut active_flat, &mut self.anchor_weights, &aggregator);
        }

        if is_sync_step {
            self.local_sequence += 1;
            if let Some(delta) = generate_local_delta(
                &active_flat,
                &mut self.anchor_weights,
                self.config.delta_selection_ratio,
                my_id,
                self.local_sequence,
            ) {
                info!(
                    step = %step_count,
                    seq = %self.local_sequence,
                    num_indices = %delta.indices.len(),
                    "Generated local delta"
                );
                relay_updates.insert(my_id, delta);
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

        if has_peer_updates {
            Ok(apply_params(model, &active_flat))
        } else {
            Ok(model)
        }
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
