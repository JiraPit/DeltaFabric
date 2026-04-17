//! PyTorch compatibility layer for DeltaFabric.
//!
//! Provides a Python-native API for DeltaFabric's distributed weight synchronization
//! using PyTorch models. Built with PyO3 for seamless Python integration.

use crate::core::{
    access_archived_packet,
    config::Config as CoreConfig,
    networking::{Node, Session},
    packet::{DeltaPacket, SparseDelta},
    sync::{apply_deltas, generate_local_delta, process_deltas},
};
use anyhow::Context;
use anyhow::Result;
use pyo3::{
    Bound, Py, PyAny, PyResult, Python, exceptions, pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Configuration for DeltaFabric's weight synchronization protocol.
///
/// # Fields
///
/// * `alpha` - Blend factor for remote deltas (default: 0.5)
/// * `delta_selection_ratio` - Fraction of changed weights to broadcast (default: 0.01)
/// * `sync_interval` - Broadcast delta every N steps (default: 100)
/// * `relay_threshold` - Minimum delta magnitude to relay (default: 1e-6)
/// * `peers` - List of peer node IDs to synchronize with
#[pyclass]
#[derive(Clone)]
pub struct Config {
    /// Blend factor for remote deltas.
    #[pyo3(get, set)]
    pub alpha: f32,
    /// Fraction of changed weights to include in delta.
    #[pyo3(get, set)]
    pub delta_selection_ratio: f32,
    /// Steps between delta broadcasts.
    #[pyo3(get, set)]
    pub sync_interval: u64,
    /// Minimum delta magnitude to relay.
    #[pyo3(get, set)]
    pub relay_threshold: f32,
    /// Peer node IDs for synchronization.
    #[pyo3(get, set)]
    pub peers: Vec<u64>,
}

#[pymethods]
impl Config {
    /// Creates a new Config with sensible defaults.
    ///
    /// # Arguments
    ///
    /// * `peers` - List of peer node IDs to synchronize with
    /// * `alpha` - Blend factor for remote deltas (default: 0.1)
    /// * `delta_selection_ratio` - Fraction of changed weights to broadcast (default: 0.01)
    /// * `sync_interval` - Broadcast delta every N steps (default: 100)
    /// * `relay_threshold` - Minimum delta magnitude to relay (default: 1e-6)
    #[new]
    #[pyo3(signature = (peers, alpha = 0.1, delta_selection_ratio = 0.01, sync_interval = 100, relay_threshold = 1e-6))]
    pub fn new(
        peers: Vec<u64>,
        alpha: f32,
        delta_selection_ratio: f32,
        sync_interval: u64,
        relay_threshold: f32,
    ) -> Self {
        Self {
            alpha,
            delta_selection_ratio,
            sync_interval,
            relay_threshold,
            peers,
        }
    }
}

impl From<Config> for CoreConfig {
    fn from(config: Config) -> Self {
        CoreConfig {
            alpha: config.alpha,
            delta_selection_ratio: config.delta_selection_ratio,
            sync_interval: config.sync_interval,
            relay_threshold: config.relay_threshold,
            peers: config.peers,
        }
    }
}

/// Manages distributed model synchronization across nodes using delta compression.
///
/// Fabric coordinates parameter updates between multiple nodes by:
/// - Aggregating incoming deltas from peers
/// - Applying synced parameters to local models
/// - Broadcasting local deltas at configurable intervals
///
/// # Example
///
/// ```python
/// from delta_fabric import Fabric, Config
///
/// config = Config(peers=[2, 3])
/// fabric = Fabric(node_id=1, config=config)
///
/// for batch in dataloader:
///     optimizer.step()
///     model = fabric.step(model)
///
/// fabric.close()
/// ```
#[pyclass]
pub struct Fabric {
    inner: Arc<Mutex<FabricInner>>,
    runtime: tokio::runtime::Runtime,
}

struct FabricInner {
    session: Session,
    config: CoreConfig,
    anchor_weights: Vec<f32>,
    seen_table: HashMap<u64, u64>,
    local_sequence: u64,
    step_count: u64,
}

impl Fabric {
    fn init_sync(node_id: u64, config: CoreConfig) -> Result<Self> {
        info!(node_id = %node_id, "Initializing DeltaFabric (PyTorch)");

        let node = Node {
            id: node_id,
            peers: config.peers.clone(),
        };

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .context("Failed to create Tokio runtime")?;

        let mut session = runtime.block_on(async { Session::new(node).await })?;
        runtime.block_on(async { session.init_fabric().await })?;

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        let inner = FabricInner {
            session,
            config,
            anchor_weights: Vec::new(),
            seen_table: HashMap::new(),
            local_sequence: 0,
            step_count: 0,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            runtime,
        })
    }

    fn step_sync(
        &self,
        params: Vec<f32>,
        keys: Vec<String>,
        shapes: Vec<Vec<i64>>,
    ) -> PyResult<Py<PyDict>> {
        #[allow(clippy::await_holding_lock)]
        let result = self.runtime.block_on(async {
            let mut inner = self.inner.lock().map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
            })?;

            let FabricInner {
                ref mut session,
                ref mut config,
                ref mut anchor_weights,
                ref mut seen_table,
                ref mut local_sequence,
                ref mut step_count,
            } = *inner;

            *step_count += 1;
            let step_count_val = *step_count;

            let mut active_flat = params;

            if anchor_weights.is_empty() {
                *anchor_weights = active_flat.clone();
                info!(
                    step = %step_count_val,
                    num_weights = %active_flat.len(),
                    "Initialized anchor weights"
                );
            }

            let mut aggregator: HashMap<u32, f32> = HashMap::new();
            let mut relay_updates: HashMap<u64, SparseDelta> = HashMap::new();

            for sample in session.pull_packets() {
                let payload = sample.payload().to_bytes();
                match access_archived_packet(&payload) {
                    Ok(incoming) => {
                        info!(node_id = %session.node.id, "Received delta packet");
                        if let Some(updates) = process_deltas(
                            &mut aggregator,
                            incoming,
                            seen_table,
                            config.alpha,
                            config.relay_threshold,
                        ) {
                            relay_updates.extend(updates);
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to deserialize incoming packet");
                    }
                }
            }

            if !aggregator.is_empty() {
                info!(node_id = %session.node.id, count = %aggregator.len(), "Applying peer deltas");
            }

            apply_deltas(&mut active_flat, anchor_weights, &aggregator);

            if step_count_val.is_multiple_of(config.sync_interval) {
                *local_sequence += 1;
                if let Some(delta) = generate_local_delta(
                    &active_flat,
                    anchor_weights,
                    config.delta_selection_ratio,
                    session.node.id,
                    *local_sequence,
                ) {
                    info!(
                        step = %step_count_val,
                        seq = %*local_sequence,
                        num_indices = %delta.indices.len(),
                        "Generated local delta"
                    );
                    relay_updates.insert(session.node.id, delta);
                }
            }

            if !relay_updates.is_empty() {
                let packet = DeltaPacket {
                    updates: relay_updates,
                };
                if let Err(e) = session.broadcast(packet).await {
                    warn!(error = %e, "Failed to broadcast packet");
                }
            }

            Ok::<_, anyhow::Error>(active_flat)
        });

        match result {
            Ok(active_flat) => {
                Python::with_gil(|py| apply_params(py, &keys, &shapes, &active_flat))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(format!(
                "Step failed: {}",
                e
            ))),
        }
    }
}

#[pymethods]
impl Fabric {
    /// Creates a new Fabric instance and initializes cluster discovery.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Unique identifier for this node (1-indexed)
    /// * `config` - Configuration containing peers and sync parameters
    ///
    /// # Returns
    ///
    /// Initialized Fabric ready for synchronization
    #[new]
    pub fn new(node_id: u64, config: Config) -> PyResult<Self> {
        let core_config: CoreConfig = config.into();
        Self::init_sync(node_id, core_config)
            .map_err(|e| exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Performs one synchronization step, updating model with peer deltas.
    ///
    /// This method:
    /// 1. Extracts parameters from the model state_dict
    /// 2. Pulls and processes incoming delta packets from peers
    /// 3. Applies aggregated deltas to model weights
    /// 4. Generates and broadcasts local delta if sync interval reached
    ///
    /// # Arguments
    ///
    /// * `model` - PyTorch nn.Module to sync (modified in-place)
    ///
    /// # Returns
    ///
    /// Updated model with synced parameters applied
    pub fn step(&self, model: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (params, shapes, keys) = extract_params(model)?;

        let py_dict = self.step_sync(params, keys, shapes)?;

        let load_method = model.getattr("load_state_dict").map_err(|e| {
            exceptions::PyAttributeError::new_err(format!("Failed to get load_state_dict: {}", e))
        })?;

        load_method.call1((py_dict,)).map_err(|e| {
            exceptions::PyRuntimeError::new_err(format!("Failed to load state_dict: {}", e))
        })?;

        Ok(model.clone().unbind())
    }

    /// Shuts down the Fabric, closing all network connections.
    pub fn close(&self) -> PyResult<()> {
        #[allow(clippy::await_holding_lock)]
        self.runtime.block_on(async {
            let mut inner = self.inner.lock().map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
            })?;

            info!(node_id = %inner.session.node.id, "Shutting down DeltaFabric");
            inner.session.shutdown().await.map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Shutdown failed: {}", e))
            })?;
            info!(node_id = %inner.session.node.id, "DeltaFabric shutdown complete");
            Ok(())
        })
    }
}

/// Result type for parameter extraction.
///
/// Contains the flattened parameters, shapes, and keys from a model's state_dict.
type ExtractResult = (Vec<f32>, Vec<Vec<i64>>, Vec<String>);

/// Extracts parameters from a PyTorch model's state_dict.
///
/// # Arguments
///
/// * `model` - PyTorch nn.Module with a state_dict() method
///
/// # Returns
///
/// Tuple of (params, shapes, keys):
/// - params: Flattened parameter values as Vec<f32>
/// - shapes: Original tensor shapes for reconstruction
/// - keys: Parameter names from state_dict
fn extract_params(model: &Bound<'_, PyAny>) -> PyResult<ExtractResult> {
    let state_dict = model.getattr("state_dict").map_err(|e| {
        exceptions::PyAttributeError::new_err(format!("Failed to get state_dict: {}", e))
    })?;

    let py_dict = state_dict.call0().map_err(|e| {
        exceptions::PyRuntimeError::new_err(format!("Failed to call state_dict(): {}", e))
    })?;

    let dict: &Bound<'_, PyDict> = py_dict.downcast().map_err(|e| {
        exceptions::PyTypeError::new_err(format!("state_dict is not a dict: {}", e))
    })?;

    let mut params = Vec::new();
    let mut shapes = Vec::new();
    let mut keys = Vec::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract().map_err(|e| {
            exceptions::PyTypeError::new_err(format!("Failed to extract key: {}", e))
        })?;
        keys.push(key_str);

        let shape_obj = value
            .getattr("shape")
            .map_err(|e| {
                exceptions::PyAttributeError::new_err(format!("Failed to get shape: {}", e))
            })?;
        let shape: Vec<i64> = shape_obj.extract().map_err(|e| {
            exceptions::PyTypeError::new_err(format!("Failed to extract shape: {}", e))
        })?;
        shapes.push(shape);

        let numpy = value
            .call_method0("detach")
            .map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Failed to call detach: {}", e))
            })?
            .call_method0("numpy")
            .map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Failed to call numpy: {}", e))
            })?
            .call_method0("flatten")
            .map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Failed to call flatten: {}", e))
            })?;
        let data: Vec<f32> = numpy.extract().map_err(|e| {
            exceptions::PyTypeError::new_err(format!("Failed to extract numpy data: {}", e))
        })?;
        params.extend(data);
    }

    Ok((params, shapes, keys))
}

/// Applies flattened parameters back to a state_dict.
///
/// # Arguments
///
/// * `keys` - Parameter names
/// * `shapes` - Original tensor shapes
/// * `params` - Flattened parameter values
///
/// # Returns
///
/// PyDict containing PyTorch tensors with updated values
fn apply_params(
    py: Python<'_>,
    keys: &[String],
    shapes: &[Vec<i64>],
    params: &[f32],
) -> PyResult<Py<PyDict>> {
    let py_dict = PyDict::new(py);
    let torch = py.import("torch")?;
    let numpy = py.import("numpy")?;

    let mut pos = 0;
    for (key, shape) in keys.iter().zip(shapes.iter()) {
        let num_elements: usize = shape.iter().map(|&x| x as usize).product();
        let end_pos = (pos + num_elements).min(params.len());
        let slice = &params[pos..end_pos];

        let np_array = numpy.call_method1("array", (slice,))?;
        let reshaped = np_array.call_method1("reshape", (shape,))?;
        let py_tensor = torch.call_method1("from_numpy", (reshaped,))?;
        let cloned_tensor = py_tensor.call_method0("clone")?;

        py_dict.set_item(key, cloned_tensor)?;
        pos = end_pos;
    }

    Ok(Py::from(py_dict))
}
