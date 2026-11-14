use anyhow::{Context, Result};

/// Enables DeltaFabric logging on stderr.
///
/// The PyTorch Fabric installs a default logger (warnings only, or the
/// RUST_LOG variable if set) when none exists, so call this before creating it.
///
/// # Arguments
///
/// * `level` - Level or filter, e.g. "info", "debug" or "warn,delta_fabric=info"
///
/// # Errors
///
/// Returns an error if the filter is invalid or logging is already initialized.
pub fn init_logging(level: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_new(level).context("Invalid log filter")?;
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .try_init()
        .map_err(|_| anyhow::anyhow!("Logging is already initialized"))
}

#[cfg(test)]
mod test {
    use super::init_logging;

    #[test]
    fn test_invalid_filter_is_rejected() {
        let err = init_logging("delta_fabric=loud").unwrap_err();
        assert!(format!("{err:#}").contains("Invalid log filter"));
    }
}
