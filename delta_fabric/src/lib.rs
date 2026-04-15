pub mod burn;
pub mod core;

use anyhow::{Context, Result};
pub use burn::Fabric;
pub use core::{
    ArchivedFabricPacketType, FabricConfig, FabricPacket, Node, NodeState, Session, SparseDelta,
    process_deltas,
};
pub mod prelude {
    pub use anyhow::{Context, Result};
}

pub fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,delta_fabric=debug"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

pub fn init_tracing_with_filter(filter: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from(filter).context("Invalid tracing filter")?;

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    Ok(())
}
