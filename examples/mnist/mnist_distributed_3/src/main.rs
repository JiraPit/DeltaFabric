use std::env;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::{
    data::dataset::{vision::MnistDataset, Dataset},
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_ndarray::{NdArray, NdArrayDevice};
use delta_fabric::{Config, Fabric};
use mnist_shared::{MnistBatch, Model};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.01;

const NUM_NODES: usize = 3;
const TRAIN_SAMPLES: usize = 60000;

pub fn init_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,delta_fabric=debug"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

fn parse_peers(peers_str: &str) -> Vec<u64> {
    peers_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

fn get_partition_range(node_id: u64) -> (usize, usize) {
    let samples_per_node = TRAIN_SAMPLES / NUM_NODES;
    let start = ((node_id - 1) as usize) * samples_per_node;
    let end = start + samples_per_node;
    (start, end)
}

pub fn load_batch<B: Backend>(
    dataset: &MnistDataset,
    partition_start: usize,
    partition_end: usize,
    batch_idx: usize,
    device: &B::Device,
) -> Option<MnistBatch<B>> {
    let start = partition_start + batch_idx * BATCH_SIZE;
    if start >= partition_end {
        return None;
    }

    let end = (start + BATCH_SIZE).min(partition_end);
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

pub async fn train<B: AutodiffBackend>(
    mut model: Model<B>,
    dataset: &MnistDataset,
    partition_start: usize,
    partition_end: usize,
    device: &B::Device,
    fabric: &mut Fabric,
) -> Result<(Model<B>, f64)> {
    let samples_per_node = partition_end - partition_start;
    let num_batches = samples_per_node / BATCH_SIZE;
    let mut total_loss: f64 = 0.0;
    let mut optimizer = SgdConfig::new().init();

    for batch_idx in 0..num_batches {
        let batch =
            load_batch::<B>(dataset, partition_start, partition_end, batch_idx, device).unwrap();

        let output = model.forward_classification(batch.clone());
        let loss = output.loss;
        total_loss += loss.to_data().into_vec::<f32>().unwrap()[0] as f64;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(LEARNING_RATE, model, grads);

        model = fabric.step(model).await.unwrap();
    }

    let avg_loss = total_loss / num_batches as f64;
    tracing::info!(loss = %avg_loss, "Epoch training complete");
    Ok((model, avg_loss))
}

pub fn accuracy<B: Backend>(
    model: &Model<B>,
    dataset: &MnistDataset,
    partition_start: usize,
    partition_end: usize,
    device: &B::Device,
) -> f64 {
    let samples_per_node = partition_end - partition_start;
    let num_batches = samples_per_node / BATCH_SIZE;
    let mut correct = 0usize;

    for batch_idx in 0..num_batches {
        let batch =
            load_batch::<B>(dataset, partition_start, partition_end, batch_idx, device).unwrap();
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

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let node_id: u64 = env::var("DF_NODE_ID")
        .expect("DF_NODE_ID must be set")
        .parse()
        .expect("DF_NODE_ID must be a valid u64");

    if node_id < 1 || node_id > NUM_NODES as u64 {
        anyhow::bail!("DF_NODE_ID must be between 1 and {}", NUM_NODES);
    }

    let peers_str = env::var("DF_PEERS").unwrap_or_default();
    let peers = if peers_str.is_empty() {
        vec![]
    } else {
        parse_peers(&peers_str)
    };

    let (partition_start, partition_end) = get_partition_range(node_id);
    let partition_size = partition_end - partition_start;

    tracing::info!(
        node_id = %node_id,
        peers = ?peers,
        partition = ?(partition_start, partition_end),
        "Starting distributed MNIST training (3 nodes)"
    );

    tracing::info!("Initializing DeltaFabric...");

    let config = Config::new(peers);
    let mut fabric = Fabric::new(node_id, config)
        .await
        .expect("Failed to initialize DeltaFabric");

    tracing::info!(node_id = %node_id, "DeltaFabric initialized");

    let device = NdArrayDevice::default();

    let mut model: Model<Autodiff<NdArray<f32>>> = Model::new(&device);
    tracing::info!(num_params = %model.num_params(), "Model initialized");

    tracing::info!("Loading MNIST data...");
    let test_dataset = MnistDataset::test();

    tracing::info!(
        test_samples = %test_dataset.len(),
        partition_samples = %partition_size,
        "Datasets loaded"
    );

    tracing::info!("Starting training with DeltaFabric sync...");

    for epoch in 0..EPOCHS {
        let dataset = MnistDataset::train();

        let (new_model, _) = train::<Autodiff<NdArray<f32>>>(
            model,
            &dataset,
            partition_start,
            partition_end,
            &device,
            &mut fabric,
        )
        .await?;
        model = new_model;

        let acc = accuracy(
            &model,
            &test_dataset,
            partition_start,
            partition_end,
            &device,
        );

        tracing::info!(
            epoch = %epoch,
            node_id = %node_id,
            accuracy = %acc,
            "Epoch complete"
        );
    }

    tracing::info!(node_id = %node_id, "Training complete");

    fabric.shutdown().await?;

    tracing::info!(node_id = %node_id, "Shutdown complete");

    Ok(())
}
