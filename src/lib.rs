// Without a framework feature nothing uses the core yet.
#[cfg_attr(
    not(any(feature = "burn", feature = "pytorch")),
    allow(dead_code, unused_imports)
)]
mod core;
mod logging;

#[cfg(all(feature = "burn", feature = "pytorch"))]
compile_error!(
    "Features 'burn' and 'pytorch' are mutually exclusive and cannot be enabled together."
);

#[cfg(feature = "burn")]
mod burn;

#[cfg(feature = "pytorch")]
mod pytorch;

pub use core::{Config, TrafficStats, ValueEncoding};
pub use logging::init_logging;

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
    m.add_function(wrap_pyfunction!(pytorch::py_init_logging, m)?)?;
    Ok(())
}
