#[cfg(test)]
mod burn_tests {
    use crate::burn::{Fabric, apply_params, extract_params};
    use crate::core::codec::ValueEncoding;
    use crate::core::config::Config;
    use burn::nn::Linear;
    use burn::nn::LinearConfig;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = burn::backend::Autodiff<NdArray<f32>>;

    fn create_test_model<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(3, 2).init(device)
    }

    fn create_test_config() -> Config {
        Config::default()
    }

    // Node IDs are unique per test and far from the IDs used by the examples,
    // because all Zenoh sessions on a host discover each other.

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
            let mut fabric = Fabric::new(2001, config).await.unwrap();

            let original_params = extract_params(&model);

            let result = fabric.step(model).await.unwrap();

            let result_params = extract_params(&result);
            assert_eq!(result_params, original_params);
            assert_eq!(fabric.session.stats().packets_sent, 0);
        });
    }

    #[test]
    fn test_anchor_weights_initialization() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(2002, config).await.unwrap();
            assert!(!fabric.state.is_initialized());

            let params = extract_params(&model);
            let _result = fabric.step(model).await.unwrap();

            assert!(fabric.state.is_initialized());
            assert_eq!(fabric.state.anchor(), params.as_slice());
            assert_eq!(fabric.state.mix_sum(), params.as_slice());
        });
    }

    #[test]
    fn test_step_count_increments() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(2003, config).await.unwrap();
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
            let mut fabric = Fabric::new(2004, config).await.unwrap();
            assert_eq!(fabric.state.local_sequence(), 0);

            let model = fabric.step(model).await.unwrap();
            assert_eq!(fabric.state.local_sequence(), 0);

            // Unchanged weights: nothing to send, sequence stays put.
            let model = fabric.step(model).await.unwrap();
            assert_eq!(fabric.state.local_sequence(), 0);

            let changed: Vec<f32> = extract_params(&model).iter().map(|p| p + 1.0).collect();
            let model = apply_params(model, &changed);
            let model = fabric.step(model).await.unwrap();
            let _ = fabric.step(model).await.unwrap();
            assert_eq!(fabric.state.local_sequence(), 1);
            assert_eq!(fabric.session.stats().packets_sent, 1);
        });
    }

    #[test]
    fn test_apply_params_preserves_gradient_tracking() {
        let device = <TestAutodiffBackend as Backend>::Device::default();

        // Create model with autodiff backend
        let model: Linear<TestAutodiffBackend> = LinearConfig::new(3, 2).init(&device);

        // Get initial parameters
        let params_before = extract_params(&model);

        // Create input tensor using from_data
        let input_data = burn::tensor::TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 3]);
        let input = Tensor::<TestAutodiffBackend, 2>::from_data(input_data, &device);

        // Forward + backward pass
        let output = model.forward(input.clone());
        let loss = output.sum();
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);

        // Apply optimizer step
        let mut optimizer = SgdConfig::new().init();
        let model = optimizer.step(0.01, model, grads_params);

        // Check that parameters actually changed (gradients were applied)
        let params_after = extract_params(&model);
        let params_changed = params_before
            .iter()
            .zip(params_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            params_changed,
            "Parameters should change after optimizer step before apply_params"
        );

        // Apply identity transform (same weights)
        let params = extract_params(&model);
        let updated_model = apply_params(model, &params);

        // Forward + backward again
        let output2 = updated_model.forward(input.clone());
        let loss2 = output2.sum();
        let grads2 = loss2.backward();
        let grads_params2 = GradientsParams::from_grads(grads2, &updated_model);

        // Apply optimizer step again
        let mut optimizer2 = SgdConfig::new().init();
        let final_model = optimizer2.step(0.01, updated_model, grads_params2);

        // Check that parameters still change after apply_params (gradients still flow)
        let final_params = extract_params(&final_model);
        let final_params_changed = params
            .iter()
            .zip(final_params.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            final_params_changed,
            "Parameters should still change after optimizer step following apply_params"
        );
    }

    #[test]
    fn test_training_with_apply_params() {
        let device = <TestAutodiffBackend as Backend>::Device::default();

        let mut model: Linear<TestAutodiffBackend> = LinearConfig::new(3, 2).init(&device);
        let mut optimizer = SgdConfig::new().init();
        let learning_rate = 0.01;

        // Create input and target tensors using from_data
        let input_data = burn::tensor::TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            [4, 3],
        );
        let target_data =
            burn::tensor::TensorData::new(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [4, 2]);
        let input = Tensor::<TestAutodiffBackend, 2>::from_data(input_data, &device);
        let targets = Tensor::<TestAutodiffBackend, 2>::from_data(target_data, &device);

        let mut losses = Vec::new();
        let two = Tensor::<TestAutodiffBackend, 2>::from_data(
            burn::tensor::TensorData::new(vec![2.0f32], [1, 1]),
            &device,
        );

        for _step in 0..10 {
            let output = model.forward(input.clone());
            // Use mean squared error manually
            let diff = output.sub(targets.clone());
            let loss = diff.powf(two.clone()).mean();
            losses.push(loss.to_data().into_vec::<f32>().unwrap()[0]);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads_params);

            // Simulate fabric.step() - apply_params with current weights
            let params = extract_params(&model);
            model = apply_params(model, &params);
        }

        // Loss should decrease
        assert!(
            losses[0] > losses[losses.len() - 1],
            "Loss should decrease during training with apply_params. Initial: {}, Final: {}",
            losses[0],
            losses[losses.len() - 1]
        );
    }

    #[test]
    fn test_new_rejects_invalid_config() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let result = Fabric::new(2005, create_test_config().gamma(2.0)).await;
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_isolated_sync_step_keeps_model() {
        // With no neighbors the mixing step must not move the weights.
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);
        let config = create_test_config().sync_interval(1).gamma(1.0);

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut fabric = Fabric::new(2006, config).await.unwrap();
            let model = fabric.step(model).await.unwrap();

            let trained: Vec<f32> = extract_params(&model).iter().map(|p| p * 0.5).collect();
            let model = apply_params(model, &trained);
            let model = fabric.step(model).await.unwrap();

            assert_eq!(extract_params(&model), trained);
        });
    }

    #[test]
    fn test_two_fabrics_pull_models_together() {
        two_fabrics_pull_models_together(2007, 2008, ValueEncoding::F32);
    }

    #[test]
    fn test_two_fabrics_pull_models_together_with_sign_values() {
        two_fabrics_pull_models_together(2009, 2010, ValueEncoding::Sign);
    }

    fn two_fabrics_pull_models_together(id_a: u64, id_b: u64, encoding: ValueEncoding) {
        let device = <TestBackend as Backend>::Device::default();
        let start = create_test_model::<TestBackend>(&device);
        let init = extract_params(&start);
        let config = |peer| {
            create_test_config()
                .peers(vec![peer])
                .sync_interval(1)
                .delta_selection_ratio(1.0)
                .gamma(0.5)
                .value_encoding(encoding)
        };

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (a, b) = tokio::join!(
                Fabric::new(id_a, config(id_b)),
                Fabric::new(id_b, config(id_a))
            );
            let (mut a, mut b) = (a.unwrap(), b.unwrap());

            // Same start, then different local "training".
            let mut model_a = a.step(apply_params(start.clone(), &init)).await.unwrap();
            let mut model_b = b.step(apply_params(start, &init)).await.unwrap();
            let pushed_a: Vec<f32> = init.iter().map(|p| p + 1.0).collect();
            let pushed_b: Vec<f32> = init.iter().map(|p| p - 1.0).collect();
            model_a = apply_params(model_a, &pushed_a);
            model_b = apply_params(model_b, &pushed_b);

            let distance = |a: &Vec<f32>, b: &Vec<f32>| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            };
            let before = distance(&extract_params(&model_a), &extract_params(&model_b));

            for _ in 0..80 {
                model_a = a.step(model_a).await.unwrap();
                model_b = b.step(model_b).await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }

            let params_a = extract_params(&model_a);
            let params_b = extract_params(&model_b);
            assert!(distance(&params_a, &params_b) < 0.01 * before);
            assert!(!a.state.is_resyncing() && !b.state.is_resyncing());

            // Mixing preserves the average of the two models. Nodes step
            // independently, so a round where one node mixes before the other's
            // delta arrives shifts it slightly (exact only in lockstep).
            for ((x, y), p) in params_a.iter().zip(&params_b).zip(&init) {
                assert!(((x + y) / 2.0 - p).abs() < 0.25);
            }
        });
    }
}
