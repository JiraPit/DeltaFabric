# DeltaFabric MNIST Example

Distributed MNIST training demonstrating DeltaFabric's weight synchronization protocol.

## Structure

```
examples/mnist/
├── Cargo.toml              # Workspace manifest
├── mnist_shared/           # Shared CNN model
│   └── src/lib.rs
├── mnist_single/           # Single-node baseline (no networking)
│   └── src/main.rs
├── mnist_distributed_2/    # 2-node distributed with DeltaFabric sync
│   └── src/main.rs
└── mnist_distributed_3/    # 3-node distributed with DeltaFabric sync
    └── src/main.rs
```

## Running

### Single Node (baseline)

```bash
cd examples/mnist
cargo run -p mnist_single
```

### 2 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2 cargo run -p mnist_distributed_2

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1 cargo run -p mnist_distributed_2
```

### 3 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2,3 cargo run -p mnist_distributed_3

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1,3 cargo run -p mnist_distributed_3

# Terminal 3
DF_NODE_ID=3 DF_PEERS=1,2 cargo run -p mnist_distributed_3
```

## DeltaFabric API

```rust
use delta_fabric::{Config, Fabric};

// 1. Create config with sensible defaults
let config = Config::new(peers);
let mut fabric = Fabric::new(node_id, config).await?;

// 2. Create model (Autodiff<NdArray> for training)
let mut model: Model<Autodiff<NdArray<f32>>> = Model::new(&device);

// 3. Training loop
for batch in dataset {
    let output = model.forward_classification(batch);
    let grads = GradientsParams::from_grads(output.loss.backward(), &model);
    model = optimizer.step(lr, model, grads);

    // DeltaFabric sync - single call, returns updated model
    model = fabric.step(model).await?;
}

// 4. Shutdown
fabric.shutdown().await?;
```

## Key Points

- **Single model**: Works with `Model<Autodiff<Backend>>` directly
- **No step count**: Fabric tracks it internally
- **No extract/apply**: `fabric.step()` handles parameter extraction and application
- **Ring topology**: 1 → 2 → ... → N → 1

## Configuration Defaults

| Parameter               | Default | Description                     |
| ----------------------- | ------- | ------------------------------- |
| `alpha`                 | 0.5     | Blend factor for remote deltas  |
| `delta_selection_ratio` | 0.01    | Only sync 1% of changed weights |
| `sync_interval`         | 100     | Broadcast delta every N steps   |
| `relay_threshold`       | 1e-6    | Minimum delta to relay          |

## Data Split

### 2 Nodes

Each node trains on 50% of MNIST:
| Node | Samples |
|------|---------|
| 1 | 0 - 29,999 |
| 2 | 30,000 - 59,999 |

### 3 Nodes

Each node trains on ~33% of MNIST:
| Node | Samples |
|------|---------|
| 1 | 0 - 19,999 |
| 2 | 20,000 - 39,999 |
| 3 | 40,000 - 59,999 |
