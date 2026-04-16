pub mod burn;
pub mod core;

use anyhow::Result;
pub use burn::{Fabric, apply_params, extract_params};
pub use core::{
    ArchivedDeltaPacket, Config, DeltaPacket, Node, NodeState, Session, SparseDelta,
    access_archived_packet, process_deltas,
};
pub mod prelude {
    pub use crate::Config;
    pub use crate::apply_params;
    pub use crate::extract_params;
    pub use anyhow::{Context, Result};
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
