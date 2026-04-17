pub mod core;

#[cfg(feature = "burn")]
pub mod burn;

#[cfg(feature = "pytorch")]
pub mod pytorch;

use anyhow::Result;
pub use core::{
    ArchivedDeltaPacket, Config, DeltaPacket, Node, NodeState, Session, SparseDelta,
    access_archived_packet, process_deltas,
};

#[cfg(feature = "burn")]
pub use burn::Fabric;

#[cfg(feature = "pytorch")]
pub use pytorch::Fabric;

#[cfg(feature = "pytorch")]
use pyo3::prelude::*;

#[cfg(feature = "pytorch")]
#[pymodule]
fn delta_fabric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pytorch::Fabric>()?;
    m.add_class::<pytorch::Config>()?;
    Ok(())
}

/// Initializes tracing with default filter "info,delta_fabric=debug".
///
/// Call once at application start to enable structured logging.
pub fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,delta_fabric=debug"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

/// Initializes tracing with a custom filter string.
///
/// # Arguments
///
/// * `filter` - Tracing filter in env_logger format (e.g., "info,debug")
///
/// # Errors
///
/// Returns an error if the filter string is invalid.
pub fn init_tracing_with_filter(filter: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::from(filter);

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    Ok(())
}
