#[cfg(test)]
mod config_tests {
    use crate::core::{Config, ValueEncoding};

    #[test]
    fn test_builder_pattern() {
        let config = Config::default()
            .gamma(0.3)
            .delta_selection_ratio(0.02)
            .sync_interval(50)
            .value_encoding(ValueEncoding::Sign)
            .peers(vec![1, 2, 3]);

        assert_eq!(config.gamma, 0.3);
        assert_eq!(config.delta_selection_ratio, 0.02);
        assert_eq!(config.sync_interval, 50);
        assert_eq!(config.value_encoding, ValueEncoding::Sign);
        assert_eq!(config.peers, vec![1, 2, 3]);
    }

    #[test]
    fn test_defaults_are_valid() {
        let config = Config::new(vec![2]);

        assert_eq!(config.gamma, 0.02);
        assert!(config.gamma <= config.max_stable_gamma());
        assert_eq!(config.delta_selection_ratio, 0.01);
        assert_eq!(config.sync_interval, 100);
        assert_eq!(config.value_encoding, ValueEncoding::F32);
        assert_eq!(config.cluster, "default");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_accepts_gamma_bounds() {
        assert!(Config::default().gamma(0.0).validate().is_ok());
        assert!(Config::default().gamma(1.0).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_invalid_values() {
        assert!(Config::default().gamma(-0.1).validate().is_err());
        assert!(Config::default().gamma(1.5).validate().is_err());
        assert!(Config::default().gamma(f32::NAN).validate().is_err());
        assert!(
            Config::default()
                .delta_selection_ratio(0.0)
                .validate()
                .is_err()
        );
        assert!(
            Config::default()
                .delta_selection_ratio(1.5)
                .validate()
                .is_err()
        );
        assert!(Config::default().sync_interval(0).validate().is_err());
    }

    #[test]
    fn test_max_stable_gamma_scales_with_ratio() {
        assert_eq!(Config::default().max_stable_gamma(), 0.02);
        assert_eq!(
            Config::default()
                .delta_selection_ratio(0.1)
                .max_stable_gamma(),
            0.2
        );
        assert_eq!(
            Config::default()
                .delta_selection_ratio(0.8)
                .max_stable_gamma(),
            1.0
        );
    }

    #[test]
    fn test_validate_cluster_names() {
        for good in ["default", "job-1", "exp_2.run3"] {
            assert!(Config::default().cluster(good).validate().is_ok(), "{good}");
        }
        for bad in ["", "a/b", "job*", "x$y", "a b", "q?"] {
            assert!(
                Config::default().cluster(bad).validate().is_err(),
                "{bad:?}"
            );
        }
    }
}
