//! PyTorch compatibility layer for DeltaFabric.
//!
//! Provides a Python-native API for DeltaFabric's distributed weight synchronization
//! using PyTorch models. Built with PyO3 for seamless Python integration.

use crate::core::{
    codec::ValueEncoding,
    config::Config as CoreConfig,
    exchange,
    networking::{Node, Session},
    sync::SyncState,
};
use anyhow::Context;
use anyhow::Result;
use pyo3::{
    Bound, Py, PyAny, PyResult, Python, exceptions, pyclass, pyfunction, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Log filter used when Python code has not chosen one: DeltaFabric warnings
/// (resyncs, rejected packets, unstable gamma) and only Zenoh errors.
const DEFAULT_LOG_FILTER: &str = "warn,zenoh=error";

/// Installs a stderr logger for DeltaFabric with the given filter.
///
/// # Errors
///
/// Returns an error if the filter is invalid or logging is already initialized.
fn install_logging(filter: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_new(filter).context("Invalid log filter")?;
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .try_init()
        .map_err(|_| anyhow::anyhow!("Logging is already initialized"))
}

/// Enables DeltaFabric logging on stderr.
///
/// Must be called before the first Fabric is created; otherwise the Fabric
/// installs the default (warnings only, or the RUST_LOG variable if set).
///
/// # Arguments
///
/// * `level` - Level or filter, e.g. "info", "debug" or "warn,delta_fabric=info"
///
/// # Errors
///
/// Raises RuntimeError if logging is already initialized or the filter is invalid.
#[pyfunction]
#[pyo3(signature = (level = "info"))]
pub fn init_logging(level: &str) -> PyResult<()> {
    install_logging(level).map_err(|e| exceptions::PyRuntimeError::new_err(format!("{:#}", e)))
}

/// Configuration for DeltaFabric's weight synchronization protocol.
///
/// # Fields
///
/// * `gamma` - Consensus step size toward the neighbors (default: 0.02)
/// * `delta_selection_ratio` - Fraction of changed weights to broadcast (default: 0.01)
/// * `sync_interval` - Broadcast delta every N steps (default: 100)
/// * `value_encoding` - "f32", "int8" or "sign" (default: "f32")
/// * `cluster` - Name of the training job (default: "default")
/// * `peers` - List of peer node IDs to synchronize with
#[pyclass]
#[derive(Clone)]
pub struct Config {
    /// Consensus step size toward the neighbors.
    #[pyo3(get, set)]
    pub gamma: f32,
    /// Fraction of changed weights to include in delta.
    #[pyo3(get, set)]
    pub delta_selection_ratio: f32,
    /// Steps between delta broadcasts.
    #[pyo3(get, set)]
    pub sync_interval: u64,
    /// How delta values are sent: "f32", "int8" or "sign".
    #[pyo3(get, set)]
    pub value_encoding: String,
    /// Name of the training job; nodes only see nodes with the same cluster.
    #[pyo3(get, set)]
    pub cluster: String,
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
    /// * `gamma` - Consensus step size toward the neighbors (default: 0.02)
    /// * `delta_selection_ratio` - Fraction of changed weights to broadcast (default: 0.01)
    /// * `sync_interval` - Broadcast delta every N steps (default: 100)
    /// * `value_encoding` - "f32" (4 bytes), "int8" (1 byte) or "sign" (1 bit) per value
    /// * `cluster` - Name of the training job (default: "default"). Separate jobs
    ///   on one network must use different names
    ///
    /// # Errors
    ///
    /// Raises ValueError for an unknown value_encoding.
    #[new]
    #[pyo3(signature = (peers, gamma = 0.02, delta_selection_ratio = 0.01, sync_interval = 100, value_encoding = "f32", cluster = "default"))]
    pub fn new(
        peers: Vec<u64>,
        gamma: f32,
        delta_selection_ratio: f32,
        sync_interval: u64,
        value_encoding: &str,
        cluster: &str,
    ) -> PyResult<Self> {
        ValueEncoding::parse(value_encoding)
            .map_err(|e| exceptions::PyValueError::new_err(format!("{:#}", e)))?;
        Ok(Self {
            gamma,
            delta_selection_ratio,
            sync_interval,
            value_encoding: value_encoding.to_string(),
            cluster: cluster.to_string(),
            peers,
        })
    }

    /// Largest gamma that reliably converges for this delta_selection_ratio.
    pub fn max_stable_gamma(&self) -> f32 {
        CoreConfig::default()
            .delta_selection_ratio(self.delta_selection_ratio)
            .max_stable_gamma()
    }
}

impl Config {
    /// Converts to the core Config.
    ///
    /// # Errors
    ///
    /// Returns an error if value_encoding was set to an unknown name.
    fn to_core(&self) -> Result<CoreConfig> {
        Ok(CoreConfig {
            gamma: self.gamma,
            delta_selection_ratio: self.delta_selection_ratio,
            value_encoding: ValueEncoding::parse(&self.value_encoding)?,
            sync_interval: self.sync_interval,
            cluster: self.cluster.clone(),
            peers: self.peers.clone(),
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
    state: SyncState,
    step_count: u64,
}

impl Fabric {
    fn init_sync(node_id: u64, config: CoreConfig) -> Result<Self> {
        // Make warnings visible unless the user already chose a logger.
        let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_string());
        let _ = install_logging(&filter);

        info!(node_id = %node_id, "Initializing DeltaFabric (PyTorch)");

        config.validate().context("Invalid DeltaFabric config")?;
        if config.gamma > config.max_stable_gamma() {
            warn!(
                gamma = %config.gamma,
                max_stable_gamma = %config.max_stable_gamma(),
                "gamma is above the stable range for this delta_selection_ratio; nodes may diverge"
            );
        }

        let node = Node {
            id: node_id,
            peers: config.peers.clone(),
            cluster: config.cluster.clone(),
        };

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .context("Failed to create Tokio runtime")?;

        let mut session = runtime.block_on(async { Session::new(node).await })?;
        runtime.block_on(async { session.init_fabric().await })?;

        let state = SyncState::new(
            config.gamma,
            config.delta_selection_ratio,
            session.mixing_weights(),
        )
        .with_encoding(config.value_encoding);

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        let inner = FabricInner {
            session,
            config,
            state,
            step_count: 0,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            runtime,
        })
    }

    fn step_sync(&self, model_handle: Py<PyAny>) -> PyResult<StepSyncResult> {
        #[allow(clippy::await_holding_lock)]
        let result = self.runtime.block_on(async {
            let mut inner = self.inner.lock().map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
            })?;

            let FabricInner {
                ref mut session,
                ref config,
                ref mut state,
                ref mut step_count,
            } = *inner;

            *step_count += 1;
            let step_count_val = *step_count;
            let is_sync_step = step_count_val.is_multiple_of(config.sync_interval);

            let extract = || {
                Python::with_gil(|py| extract_params(model_handle.bind(py)))
                    .map_err(anyhow::Error::new)
            };

            let mut extracted = None;
            if !state.is_initialized() {
                let (active_flat, shapes, keys) = extract()?;
                state.initialize(&active_flat);
                extracted = Some((active_flat, shapes, keys));
            }

            exchange(session, state).await?;

            if !is_sync_step {
                return Ok::<_, anyhow::Error>(None);
            }

            let (mut active_flat, shapes, keys) = match extracted {
                Some(params) => params,
                None => extract()?,
            };

            if let Some(packet) = state.sync_step(&mut active_flat)? {
                info!(
                    step = %step_count_val,
                    seq = %packet.delta.sequence_id,
                    num_indices = %packet.delta.count,
                    "Broadcasting local delta"
                );
                if let Err(e) = session.broadcast(packet).await {
                    warn!(error = %e, "Failed to broadcast packet");
                }
            }

            Ok::<_, anyhow::Error>(Some((active_flat, keys, shapes)))
        });

        result.map_err(|e| exceptions::PyRuntimeError::new_err(format!("Step failed: {:#}", e)))
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
        config
            .to_core()
            .and_then(|core_config| Self::init_sync(node_id, core_config))
            .map_err(|e| exceptions::PyRuntimeError::new_err(format!("{:#}", e)))
    }

    /// Performs one synchronization step.
    ///
    /// This method:
    /// 1. Initializes the anchor from the model's floating-point state on the first step
    /// 2. Applies received peer deltas and handles resyncs (no model access)
    /// 3. On sync steps, mixes the model toward its neighbors, broadcasts
    ///    its own delta and loads the weights back into the model
    ///
    /// # Arguments
    ///
    /// * `model` - PyTorch nn.Module to sync (modified in-place)
    ///
    /// # Returns
    ///
    /// Updated model with synced parameters applied
    pub fn step(&self, model: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let model_handle = model.clone().unbind();

        if let Some((active_flat, keys, shapes)) = self.step_sync(model_handle)? {
            let py_dict = Python::with_gil(|py| apply_params(py, &keys, &shapes, &active_flat))?;

            let load_method = model.getattr("load_state_dict").map_err(|e| {
                exceptions::PyAttributeError::new_err(format!(
                    "Failed to get load_state_dict: {}",
                    e
                ))
            })?;

            // Only floating-point entries are synced, so other entries stay as they are.
            let kwargs = PyDict::new(model.py());
            kwargs.set_item("strict", false)?;
            load_method.call((py_dict,), Some(&kwargs)).map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!("Failed to load state_dict: {}", e))
            })?;
        }

        Ok(model.clone().unbind())
    }

    /// Returns delta traffic counters.
    ///
    /// # Returns
    ///
    /// Dict with delta counters (`packets_sent`, `bytes_sent`, `packets_received`,
    /// `bytes_received`) and resync snapshot counters (`snapshots_sent`,
    /// `snapshot_bytes_sent`, `snapshots_received`, `snapshot_bytes_received`).
    /// Byte counts are serialized payload sizes (excluding Zenoh framing).
    pub fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let stats = self
            .inner
            .lock()
            .map_err(|e| exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e)))?
            .session
            .stats();

        let dict = PyDict::new(py);
        dict.set_item("packets_sent", stats.packets_sent)?;
        dict.set_item("bytes_sent", stats.bytes_sent)?;
        dict.set_item("packets_received", stats.packets_received)?;
        dict.set_item("bytes_received", stats.bytes_received)?;
        dict.set_item("snapshots_sent", stats.snapshots_sent)?;
        dict.set_item("snapshot_bytes_sent", stats.snapshot_bytes_sent)?;
        dict.set_item("snapshots_received", stats.snapshots_received)?;
        dict.set_item("snapshot_bytes_received", stats.snapshot_bytes_received)?;
        Ok(Py::from(dict))
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

/// Return type for step_sync: the synced parameters on sync steps, otherwise None.
type StepSyncResult = Option<(Vec<f32>, Vec<String>, Vec<Vec<i64>>)>;

/// Extracts floating-point parameters and buffers from a PyTorch model's state_dict.
///
/// Integer entries (such as BatchNorm's num_batches_tracked) are skipped:
/// they are counters, not weights, and must not be averaged.
///
/// # Arguments
///
/// * `model` - PyTorch nn.Module with a state_dict() method
///
/// # Returns
///
/// Tuple of (params, shapes, keys):
/// - `params`: Flattened parameter values as Vec<f32>
/// - `shapes`: Original tensor shapes for reconstruction
/// - `keys`: Parameter names from state_dict
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
        let is_float: bool = value
            .call_method0("is_floating_point")
            .and_then(|v| v.extract())
            .map_err(|e| {
                exceptions::PyTypeError::new_err(format!("Failed to check tensor dtype: {}", e))
            })?;
        if !is_float {
            continue;
        }

        let key_str: String = key.extract().map_err(|e| {
            exceptions::PyTypeError::new_err(format!("Failed to extract key: {}", e))
        })?;
        keys.push(key_str);

        let shape_obj = value.getattr("shape").map_err(|e| {
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
/// * `py` - Python GIL token
/// * `keys` - Parameter names
/// * `shapes` - Original tensor shapes
/// * `params` - Flattened parameter values
///
/// # Returns
///
/// `PyDict` containing PyTorch tensors with updated values
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
