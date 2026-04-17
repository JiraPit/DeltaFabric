use anyhow::Result;
use burn::backend::Autodiff;
use burn::{
    data::dataset::{vision::MnistDataset, Dataset},
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_tch::{LibTorch, LibTorchDevice};
use mnist_shared::{MnistBatch, Model};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.01;
const TRAIN_SAMPLES: usize = 60000;

pub fn init_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

pub fn load_batch<B: Backend>(
    dataset: &MnistDataset,
    start: usize,
    device: &B::Device,
) -> Option<MnistBatch<B>> {
    if start >= dataset.len() {
        return None;
    }

    let end = (start + BATCH_SIZE).min(dataset.len());
    let mut images = Vec::new();
    let mut targets = Vec::new();

    for i in start..end {
        if let Some(item) = dataset.get(i) {
            let flat_image: Vec<f32> = item.image.iter().flatten().copied().collect();
            let img: Tensor<B, 3> = Tensor::from_data(
                burn::tensor::TensorData::new(flat_image, [1, 28, 28]),
                device,
            );
            let img = (img / 255.0 - 0.1307) / 0.3081;
            let label: Tensor<B, 1, Int> = Tensor::from_data(
                burn::tensor::TensorData::new(vec![item.label as i64], [1]),
                device,
            );
            images.push(img);
            targets.push(label);
        }
    }

    if images.is_empty() {
        return None;
    }

    let images = Tensor::cat(images, 0);
    let targets = Tensor::cat(targets, 0);

    Some(MnistBatch { images, targets })
}

pub fn train<B: AutodiffBackend>(
    model: &mut Model<B>,
    dataset: &MnistDataset,
    device: &B::Device,
    num_epochs: usize,
) {
    let config = SgdConfig::new();
    let mut optimizer = config.init();

    for epoch in 0..num_epochs {
        let num_batches = TRAIN_SAMPLES / BATCH_SIZE;
        let mut total_loss: f64 = 0.0;

        for batch_idx in 0..num_batches {
            let batch = load_batch::<B>(dataset, batch_idx * BATCH_SIZE, device).unwrap();

            let output = model.forward_classification(batch.clone());
            let loss = output.loss;
            total_loss += loss.to_data().into_vec::<f32>().unwrap()[0] as f64;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            *model = optimizer.step(LEARNING_RATE, model.clone(), grads);
        }

        let avg_loss = total_loss / num_batches as f64;
        tracing::info!(epoch = %epoch, loss = %avg_loss, "Epoch complete");
    }
}

pub fn accuracy<B: Backend>(model: &Model<B>, dataset: &MnistDataset, device: &B::Device) -> f64 {
    let num_batches = dataset.len() / BATCH_SIZE;
    let mut correct = 0usize;

    for batch_idx in 0..num_batches {
        let batch = load_batch::<B>(dataset, batch_idx * BATCH_SIZE, device).unwrap();
        let output = model.forward_classification(batch.clone());

        let predictions = output.output.argmax(1);
        let targets = batch.targets;

        let pred_vec = predictions.to_data().to_vec::<i64>().unwrap();
        let target_vec = targets.to_data().to_vec::<i64>().unwrap();

        for (p, t) in pred_vec.iter().zip(target_vec.iter()) {
            if *p == *t {
                correct += 1;
            }
        }
    }

    let total = num_batches * BATCH_SIZE;
    correct as f64 / total as f64
}

fn main() -> Result<()> {
    init_tracing();

    let device = LibTorchDevice::default();

    tracing::info!("Initializing single-node MNIST training");

    tracing::info!(
        "Epochs: {}, Batch size: {}, LR: {}",
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE
    );

    let mut model: Model<Autodiff<LibTorch<f32>>> = Model::new(&device);
    tracing::info!(num_params = %model.num_params(), "Model initialized");

    tracing::info!("Loading MNIST data...");
    let train_dataset = MnistDataset::train();
    let test_dataset = MnistDataset::test();

    tracing::info!(
        train_samples = %TRAIN_SAMPLES,
        train_available = %train_dataset.len(),
        test_samples = %test_dataset.len(),
        "Datasets loaded"
    );

    tracing::info!("Starting training...");

    train(&mut model, &train_dataset, &device, EPOCHS);

    tracing::info!("Training complete");

    let acc = accuracy(&model, &test_dataset, &device);
    tracing::info!(accuracy = %acc, "Final test accuracy");

    Ok(())
}
