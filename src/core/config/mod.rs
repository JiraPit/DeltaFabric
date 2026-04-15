#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub alpha: f32,
    pub delta_selection_ratio: f32,
    pub sync_interval: u64,
    pub relay_threshold: f32,
    pub peers: Vec<u64>,
}

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
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn delta_selection_ratio(mut self, delta_selection_ratio: f32) -> Self {
        self.delta_selection_ratio = delta_selection_ratio;
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

    pub fn peers(mut self, peers: Vec<u64>) -> Self {
        self.peers = peers;
        self
    }
}

#[cfg(test)]
mod test;
