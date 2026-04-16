/// Configuration parameters for DeltaFabric synchronization.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Damping factor applied to incoming deltas (0.0 to 1.0). Higher values
    /// cause faster adaptation but less stability.
    pub alpha: f32,
    /// Fraction of parameter indices to include in delta broadcasts.
    /// Higher values increase communication but may improve convergence.
    pub delta_selection_ratio: f32,
    /// Number of steps between local delta broadcasts. Higher values
    /// reduce network traffic but may slow synchronization.
    pub sync_interval: u64,
    /// Minimum absolute delta value to relay to peers. Smaller values
    /// are filtered out to reduce noise.
    pub relay_threshold: f32,
    /// List of peer node IDs to communicate with.
    pub peers: Vec<u64>,
}

/// Creates a Config with default values.
impl Default for Config {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            delta_selection_ratio: 0.01,
            sync_interval: 100,
            relay_threshold: 1e-6,
            peers: vec![],
        }
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
    /// - alpha: 0.5
    /// - delta_selection_ratio: 0.01
    /// - sync_interval: 100
    /// - relay_threshold: 1e-6
    pub fn new(peers: Vec<u64>) -> Self {
        Self {
            alpha: 0.5,
            delta_selection_ratio: 0.01,
            sync_interval: 100,
            relay_threshold: 1e-6,
            peers,
        }
    }

    /// Sets the damping factor. Returns self for method chaining.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the delta selection ratio. Returns self for method chaining.
    pub fn delta_selection_ratio(mut self, delta_selection_ratio: f32) -> Self {
        self.delta_selection_ratio = delta_selection_ratio;
        self
    }

    /// Sets the sync interval. Returns self for method chaining.
    pub fn sync_interval(mut self, sync_interval: u64) -> Self {
        self.sync_interval = sync_interval;
        self
    }

    /// Sets the relay threshold. Returns self for method chaining.
    pub fn relay_threshold(mut self, relay_threshold: f32) -> Self {
        self.relay_threshold = relay_threshold;
        self
    }

    /// Sets the peer list. Returns self for method chaining.
    pub fn peers(mut self, peers: Vec<u64>) -> Self {
        self.peers = peers;
        self
    }
}

#[cfg(test)]
mod test;
