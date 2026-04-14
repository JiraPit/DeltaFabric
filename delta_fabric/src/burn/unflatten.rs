use anyhow::Result;
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use tracing::{debug, warn};

struct ParamSetter {
    data: Vec<f32>,
    pos: usize,
}

impl<B: Backend> ModuleMapper<B> for ParamSetter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let num_elements = param.val().dims().iter().product::<usize>();
        let end_pos = self.pos + num_elements;

        if end_pos > self.data.len() {
            warn!(
                expected = %num_elements,
                available = %(self.data.len() - self.pos),
                "Insufficient data for parameter"
            );
        }

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

pub fn unflatten_burn_model<B: Backend, M: Module<B>>(model: M, flat_data: &[f32]) -> Result<M> {
    let num_params = model.num_params();
    debug!(num_params = %num_params, "Unflattening model parameters");

    let mut setter = ParamSetter {
        data: flat_data.to_vec(),
        pos: 0,
    };

    let model = model.map(&mut setter);

    debug!(
        pos = %setter.pos,
        expected = %flat_data.len(),
        "Unflatten complete"
    );

    if setter.pos != flat_data.len() {
        warn!(
            consumed = %setter.pos,
            expected = %flat_data.len(),
            "Not all data was consumed during unflatten"
        );
    }

    Ok(model)
}
