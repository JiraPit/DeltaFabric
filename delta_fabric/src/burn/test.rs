#[cfg(test)]
mod burn_tests {
    use crate::burn::{flatten_burn_model, unflatten_burn_model};
    use burn::nn::Linear;
    use burn::nn::LinearConfig;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn create_test_model<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(3, 2).init(device)
    }

    #[test]
    fn test_flatten_unflatten_preserves_params() {
        let device = <TestBackend as Backend>::Device::default();
        let model = create_test_model::<TestBackend>(&device);

        let original_flat = flatten_burn_model(&model).unwrap();

        assert!(!original_flat.is_empty());
        assert_eq!(original_flat.len(), 3 * 2 + 2);

        let modified_flat: Vec<f32> = original_flat.iter().map(|x| x * 2.0).collect();

        let unflattened_model = unflatten_burn_model(model, &modified_flat).unwrap();

        let unflattened_flat = flatten_burn_model(&unflattened_model).unwrap();

        assert_eq!(unflattened_flat, modified_flat);
    }
}
