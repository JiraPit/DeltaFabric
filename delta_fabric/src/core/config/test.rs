#[cfg(test)]
mod test {
    use crate::core::FabricConfig;

    #[test]
    fn test_default_config() {
        let config = FabricConfig::default();

        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.top_k_pct, 0.01);
        assert_eq!(config.sync_interval, 100);
        assert_eq!(config.relay_threshold, 1e-6);
        assert!(config.expected_peers.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let config = FabricConfig::default()
            .alpha(0.3)
            .top_k_pct(0.02)
            .sync_interval(50)
            .relay_threshold(1e-5)
            .expected_peers(vec![1, 2, 3]);

        assert_eq!(config.alpha, 0.3);
        assert_eq!(config.top_k_pct, 0.02);
        assert_eq!(config.sync_interval, 50);
        assert_eq!(config.relay_threshold, 1e-5);
        assert_eq!(config.expected_peers, vec![1, 2, 3]);
    }
}
