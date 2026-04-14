pub mod apply;
pub mod generate;

pub use apply::{apply_deltas, process_deltas};
pub use generate::generate_local_delta;

#[cfg(test)]
mod test;
