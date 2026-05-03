#[cfg(test)]
mod burn_tests {
    use crate::burn::{Fabric, apply_params, extract_params};
    use crate::core::config::Config;
    use burn::nn::Linear;
    use burn::nn::LinearConfig;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn create_test_model<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(3, 2).init(device)
    }

    fn create_test_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_extract_apply_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let original_params = extract_params(&model);

        assert!(!original_params.is_empty());
        assert_eq!(original_params.len(), 3 * 2 + 2);

        let modified_params: Vec<f32> = original_params.iter().map(|x| x * 2.0).collect();

        let updated_model = apply_params(model, &modified_params);

        let reextracted = extract_params(&updated_model);

        assert_eq!(reextracted, modified_params);
    }

    #[test]
    fn test_extract_params_returns_all_params() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let params = extract_params(&model);

        assert_eq!(params.len(), 3 * 2 + 2);
    }

    #[test]
    fn test_apply_params_updates_weights() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let original_params = extract_params(&model);
        let mut new_params = original_params.clone();
        for (i, p) in new_params.iter_mut().enumerate() {
            *p = i as f32;
        }

        let updated_model = apply_params(model, &new_params);
        let extracted = extract_params(&updated_model);

        assert_eq!(extracted, new_params);
    }

    #[test]
    fn test_extract_params_preserves_order() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let params = extract_params(&model);

        let weights_len = 3 * 2;
        let bias_len = 2;

        assert_eq!(params.len(), weights_len + bias_len);
    }

    #[test]
    fn test_apply_params_with_zeros() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let zeros = vec![0.0; 3 * 2 + 2];

        let zeroed_model = apply_params(model, &zeros);
        let extracted = extract_params(&zeroed_model);

        assert_eq!(extracted, zeros);
    }

    #[test]
    fn test_different_model_sizes() {
        let device = <TestBackend as Backend>::Device::default();

        let small: Linear<TestBackend> = LinearConfig::new(10, 5).init(&device);
        let params_small = extract_params(&small);
        assert_eq!(params_small.len(), 10 * 5 + 5);

        let large: Linear<TestBackend> = LinearConfig::new(1024, 512).init(&device);
        let params_large = extract_params(&large);
        assert_eq!(params_large.len(), 1024 * 512 + 512);
    }

    #[test]
    fn test_need_active_optimization_no_activity() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(1, config).await.unwrap();
            fabric.anchor_weights = vec![0.0; 8];

            let original_params = extract_params(&model);

            let result = fabric.step(model).await.unwrap();

            let result_params = extract_params(&result);
            assert_eq!(result_params, original_params);
            assert!(fabric.seen_table.is_empty());
        });
    }

    #[test]
    fn test_anchor_weights_initialization() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(1, config).await.unwrap();
            assert!(fabric.anchor_weights.is_empty());

            let _result = fabric.step(model).await.unwrap();

            assert!(!fabric.anchor_weights.is_empty());
            assert_eq!(fabric.anchor_weights.len(), 8);
        });
    }

    #[test]
    fn test_step_count_increments() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(1, config).await.unwrap();
            assert_eq!(fabric.step_count, 0);

            let _result = fabric.step(model).await.unwrap();
            assert_eq!(fabric.step_count, 1);
        });
    }

    #[test]
    fn test_local_sequence_increments_on_sync_step() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let mut config = create_test_config();
        config.sync_interval = 2;

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(1, config).await.unwrap();
            assert_eq!(fabric.local_sequence, 0);

            let _result = fabric.step(model).await.unwrap();
            assert_eq!(fabric.local_sequence, 0);

            let model2 = create_test_model::<TestBackend>(&device);
            let _result = fabric.step(model2).await.unwrap();
            assert_eq!(fabric.local_sequence, 1);
        });
    }
}
