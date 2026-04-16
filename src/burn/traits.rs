use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::Tensor;

/// Extracts all learnable parameters from a Burn model into a flat vector.
///
/// # Arguments
///
/// * `model` - The model to extract parameters from
///
/// # Returns
///
/// Vector of f32 values containing all model parameters in order.
pub fn extract_params<M: Module<B>, B: burn::tensor::backend::Backend>(model: &M) -> Vec<f32> {
    let mut collector = ParamCollector { data: Vec::new() };
    model.visit(&mut collector);
    collector.data
}

/// Applies a flat parameter vector to a Burn model, returning the updated model.
///
/// # Arguments
///
/// * `model` - The model to update (taken by value)
/// * `params` - Flat vector of f32 values to apply
///
/// # Returns
///
/// Model with parameters replaced by the provided values.
pub fn apply_params<M: Module<B>, B: burn::tensor::backend::Backend>(
    model: M,
    params: &[f32],
) -> M {
    let mut setter = ParamSetter {
        data: params.to_vec(),
        pos: 0,
    };
    model.map(&mut setter)
}

struct ParamCollector {
    data: Vec<f32>,
}

impl<B: burn::tensor::backend::Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let values: Vec<f32> = param.val().to_data().into_vec().unwrap();
        self.data.extend(values);
    }
}

struct ParamSetter {
    data: Vec<f32>,
    pos: usize,
}

impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for ParamSetter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let num_elements = param.val().dims().iter().product::<usize>();
        let end_pos = self.pos + num_elements;
        let tensor_data = self.data[self.pos..end_pos.min(self.data.len())].to_vec();
        self.pos = end_pos;

        let shape = param.val().dims().to_vec();
        let new_tensor = Tensor::<B, D>::from_data(
            burn::tensor::TensorData::new(tensor_data, shape),
            &param.val().device(),
        );

        param.map(|_| new_tensor)
    }
}
