#[derive(Debug, Clone, PartialEq)]
pub struct FabricConfig {
    pub alpha: f32,
    pub top_k_pct: f32,
    pub sync_interval: u64,
    pub relay_threshold: f32,
    pub expected_peers: Vec<u64>,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            top_k_pct: 0.01,
            sync_interval: 100,
            relay_threshold: 1e-6,
            expected_peers: vec![],
        }
    }
}

impl FabricConfig {
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn top_k_pct(mut self, top_k_pct: f32) -> Self {
        self.top_k_pct = top_k_pct;
        self
    }

    pub fn sync_interval(mut self, sync_interval: u64) -> Self {
        self.sync_interval = sync_interval;
        self
    }

    pub fn relay_threshold(mut self, relay_threshold: f32) -> Self {
        self.relay_threshold = relay_threshold;
        self
    }

    pub fn expected_peers(mut self, expected_peers: Vec<u64>) -> Self {
        self.expected_peers = expected_peers;
        self
    }
}

#[cfg(test)]
mod test;
