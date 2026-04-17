use std::env;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::{
    data::dataset::{vision::MnistDataset, Dataset},
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_tch::{LibTorch, LibTorchDevice};
use delta_fabric::{Config, Fabric};
use mnist_shared::{MnistBatch, Model};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.01;

const NUM_NODES: usize = 2;
const TRAIN_SAMPLES: usize = 60000;
const TRAIN_SAMPLES_PER_NODE: usize = TRAIN_SAMPLES / NUM_NODES;
const SYNC_INTERVAL: usize = (TRAIN_SAMPLES_PER_NODE / BATCH_SIZE) / 2;
const SEED: u64 = 42;

fn shuffle_indices(seed: u64, count: usize) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut indices: Vec<usize> = (0..count).collect();
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let hash = hasher.finish();
    
    for i in (1..count).rev() {
        let j = ((hash.wrapping_mul((i + 1) as u64)) % (i + 1) as u64) as usize;
        indices.swap(i, j);
    }
    indices
}

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
    let start = ((node_id - 1) as usize) * TRAIN_SAMPLES_PER_NODE;
    let end = start + TRAIN_SAMPLES_PER_NODE;
    (start, end)
}

pub fn load_batch_by_indices<B: Backend>(
    dataset: &MnistDataset,
    indices: &[usize],
    batch_idx: usize,
    device: &B::Device,
) -> Option<MnistBatch<B>> {
    let start = batch_idx * BATCH_SIZE;
    if start >= indices.len() {
        return None;
    }

    let end = (start + BATCH_SIZE).min(indices.len());
    let mut images = Vec::new();
    let mut targets = Vec::new();

    for &i in &indices[start..end] {
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
    indices: &[usize],
    device: &B::Device,
    fabric: &mut Fabric,
) -> Result<(Model<B>, f64)> {
    let num_batches = indices.len() / BATCH_SIZE;
    let mut total_loss: f64 = 0.0;
    let mut optimizer = SgdConfig::new().init();

    for batch_idx in 0..num_batches {
        let batch = load_batch_by_indices::<B>(dataset, indices, batch_idx, device).unwrap();

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
    device: &B::Device,
) -> f64 {
    let num_batches = dataset.len() / BATCH_SIZE;
    let mut correct = 0usize;

    for batch_idx in 0..num_batches {
        let batch = load_batch_by_indices::<B>(dataset, &(0..dataset.len()).collect::<Vec<_>>(), batch_idx, device).unwrap();
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

    tracing::info!(
        node_id = %node_id,
        peers = ?peers,
        partition = ?(partition_start, partition_end),
        "Starting distributed MNIST training (2 nodes)"
    );

    tracing::info!("Initializing DeltaFabric...");

    let config = Config::new(peers)
        .alpha(0.1)                      // blend factor
        .delta_selection_ratio(0.01)    // 1% of weights
        .sync_interval(SYNC_INTERVAL as u64);
    let mut fabric = Fabric::new(node_id, config)
        .await
        .expect("Failed to initialize DeltaFabric");

    tracing::info!(node_id = %node_id, "DeltaFabric initialized");

    let device = LibTorchDevice::default();
    tch::manual_seed(SEED as i64);

    let mut model: Model<Autodiff<LibTorch<f32>>> = Model::new(&device);
    tracing::info!(num_params = %model.num_params(), "Model initialized");

    tracing::info!("Loading MNIST data...");
    let test_dataset = MnistDataset::test();

    tracing::info!(
        train_samples = %TRAIN_SAMPLES,
        test_samples = %test_dataset.len(),
        "Datasets loaded"
    );

    tracing::info!("Starting training with DeltaFabric sync...");

    for epoch in 0..EPOCHS {
        let dataset = MnistDataset::train();
        let shuffled_indices = shuffle_indices(SEED, TRAIN_SAMPLES);
        let my_indices: Vec<usize> = shuffled_indices
            .into_iter()
            .skip(partition_start)
            .take(TRAIN_SAMPLES_PER_NODE)
            .collect();

        let (new_model, _) = train::<Autodiff<LibTorch<f32>>>(
            model, &dataset, &my_indices, &device, &mut fabric,
        )
        .await?;
        model = new_model;

        let acc = accuracy(&model, &test_dataset, &device);

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
