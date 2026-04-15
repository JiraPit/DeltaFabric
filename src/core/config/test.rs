#[cfg(test)]
mod config_tests {
    use crate::core::Config;

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.delta_selection_ratio, 0.01);
        assert_eq!(config.sync_interval, 100);
        assert_eq!(config.relay_threshold, 1e-6);
        assert!(config.peers.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let config = Config::default()
            .alpha(0.3)
            .delta_selection_ratio(0.02)
            .sync_interval(50)
            .relay_threshold(1e-5)
            .peers(vec![1, 2, 3]);

        assert_eq!(config.alpha, 0.3);
        assert_eq!(config.delta_selection_ratio, 0.02);
        assert_eq!(config.sync_interval, 50);
        assert_eq!(config.relay_threshold, 1e-5);
        assert_eq!(config.peers, vec![1, 2, 3]);
    }
}
