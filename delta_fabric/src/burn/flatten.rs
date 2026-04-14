use anyhow::Result;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use tracing::debug;

struct ParamCollector {
    data: Vec<f32>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let values: Vec<f32> = param.val().to_data().into_vec().unwrap();
        self.data.extend(values);
    }
}

pub fn flatten_burn_model<B: Backend, M: Module<B>>(model: &M) -> Result<Vec<f32>> {
    let num_params = model.num_params();
    debug!(num_params = %num_params, "Flattening model parameters");

    let mut collector = ParamCollector { data: Vec::new() };
    model.visit(&mut collector);

    debug!(num_elements = %collector.data.len(), "Flattened model to vector");
    Ok(collector.data)
}
