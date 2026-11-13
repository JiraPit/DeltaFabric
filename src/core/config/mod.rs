use crate::core::codec::ValueEncoding;
use anyhow::{Result, ensure};

/// Configuration parameters for DeltaFabric synchronization.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Consensus step size (0.0 to 1.0). At each sync the model moves this
    /// fraction of the way toward the weighted average of its neighbors'
    /// anchors. 0.0 disables mixing (independent training). Heavier
    /// compression needs a smaller gamma: above about 2x
    /// delta_selection_ratio the nodes can diverge (see max_stable_gamma).
    pub gamma: f32,
    /// Fraction of parameter indices to include in delta broadcasts.
    /// Higher values increase communication but may improve convergence.
    pub delta_selection_ratio: f32,
    /// How delta values are sent: exact f32 (4 bytes), int8 (1 byte) or
    /// sign (1 bit). Rounding error is not lost: it stays in the local
    /// difference and is sent later.
    pub value_encoding: ValueEncoding,
    /// Number of steps between local delta broadcasts. Higher values
    /// reduce network traffic but may slow synchronization.
    pub sync_interval: u64,
    /// List of peer node IDs to communicate with. Links are undirected, so
    /// list each link in both nodes' peers. A one-sided link only works if
    /// both nodes discover each other; otherwise initialization fails.
    pub peers: Vec<u64>,
}

/// Creates a Config with default values.
impl Default for Config {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl Config {
    /// Creates a Config with sensible defaults and the specified peers.
    ///
    /// # Arguments
    ///
    /// * `peers` - List of peer node IDs
    ///
    /// # Defaults
    ///
    /// - gamma: 0.02
    /// - delta_selection_ratio: 0.01
    /// - value_encoding: F32
    /// - sync_interval: 100
    pub fn new(peers: Vec<u64>) -> Self {
        Self {
            gamma: 0.02,
            delta_selection_ratio: 0.01,
            value_encoding: ValueEncoding::F32,
            sync_interval: 100,
            peers,
        }
    }

    /// Sets the consensus step size. Returns self for method chaining.
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the delta selection ratio. Returns self for method chaining.
    pub fn delta_selection_ratio(mut self, delta_selection_ratio: f32) -> Self {
        self.delta_selection_ratio = delta_selection_ratio;
        self
    }

    /// Sets the value encoding. Returns self for method chaining.
    pub fn value_encoding(mut self, value_encoding: ValueEncoding) -> Self {
        self.value_encoding = value_encoding;
        self
    }

    /// Sets the sync interval. Returns self for method chaining.
    pub fn sync_interval(mut self, sync_interval: u64) -> Self {
        self.sync_interval = sync_interval;
        self
    }

    /// Sets the peer list. Returns self for method chaining.
    pub fn peers(mut self, peers: Vec<u64>) -> Self {
        self.peers = peers;
        self
    }

    /// Largest gamma that reliably converged for this delta_selection_ratio.
    ///
    /// Measured on mesh, ring and star topologies with top-k compression:
    /// gamma = 2 x ratio converged everywhere, 5 x ratio diverged everywhere.
    pub fn max_stable_gamma(&self) -> f32 {
        (2.0 * self.delta_selection_ratio).min(1.0)
    }

    /// Checks that all parameters are within their valid ranges.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first invalid parameter.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.0..=1.0).contains(&self.gamma),
            "gamma must be in [0, 1], got {}",
            self.gamma
        );
        ensure!(
            self.delta_selection_ratio > 0.0 && self.delta_selection_ratio <= 1.0,
            "delta_selection_ratio must be in (0, 1], got {}",
            self.delta_selection_ratio
        );
        ensure!(self.sync_interval > 0, "sync_interval must be at least 1");
        Ok(())
    }
}

#[cfg(test)]
mod test;
