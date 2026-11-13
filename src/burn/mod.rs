#[cfg(test)]
mod test;

use crate::core::{config::Config, exchange, networking::Session, sync::SyncState};
use anyhow::{Context, Result};
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
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
/// - Adding received peer deltas to the mixing sum every step
/// - At each sync interval, moving the model toward its neighbors and
///   broadcasting its own top-k anchor change
/// - Resyncing from peer snapshots when a delta is lost
pub struct Fabric {
    /// Zenoh session for network communication
    pub session: Session,
    /// Configuration parameters for delta sync
    pub config: Config,
    /// Sync protocol state (anchor, mixing sum, sequence tracking)
    pub state: SyncState,
    /// Total number of steps performed
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
    /// Returns an error if the config is invalid, or if Zenoh session
    /// creation or cluster initialization fails.
    pub async fn new(node_id: u64, config: Config) -> Result<Self> {
        info!(node_id = %node_id, "Initializing DeltaFabric");

        config.validate().context("Invalid DeltaFabric config")?;
        if config.gamma > config.max_stable_gamma() {
            warn!(
                gamma = %config.gamma,
                max_stable_gamma = %config.max_stable_gamma(),
                "gamma is above the stable range for this delta_selection_ratio; nodes may diverge"
            );
        }

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

        let state = SyncState::new(
            config.gamma,
            config.delta_selection_ratio,
            session.mixing_weights(),
        )
        .with_encoding(config.value_encoding);

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        Ok(Self {
            session,
            config,
            state,
            step_count: 0,
        })
    }

    /// Performs one synchronization step.
    ///
    /// This method:
    /// 1. Initializes the anchor from the model on the first step
    /// 2. Applies received peer deltas and handles resyncs (no model access)
    /// 3. On sync steps, mixes the model toward its neighbors, broadcasts
    ///    its own delta and writes the weights back
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
    /// The model, with synced parameters applied on sync steps.
    pub async fn step<B: Backend, M: Module<B>>(&mut self, model: M) -> Result<M> {
        self.step_count += 1;
        let is_sync_step = self.step_count.is_multiple_of(self.config.sync_interval);

        let mut active_flat = None;
        if !self.state.is_initialized() {
            let flat = extract_params(&model);
            self.state.initialize(&flat);
            active_flat = Some(flat);
        }

        exchange(&mut self.session, &mut self.state).await?;

        if !is_sync_step {
            return Ok(model);
        }

        let mut active_flat = active_flat.unwrap_or_else(|| extract_params(&model));
        if let Some(packet) = self.state.sync_step(&mut active_flat)? {
            info!(
                step = %self.step_count,
                seq = %packet.delta.sequence_id,
                num_indices = %packet.delta.count,
                "Broadcasting local delta"
            );
            self.session
                .broadcast(packet)
                .await
                .context("Failed to broadcast packet")?;
        }

        Ok(apply_params(model, &active_flat))
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
