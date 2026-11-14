# DeltaFabric PyTorch MNIST Example

Distributed MNIST training demonstrating DeltaFabric's weight synchronization protocol with PyTorch.

## Prerequisites

Install DeltaFabric:

```bash
pip install delta_fabric
```

Or build it from this repository (from the repository root):

```bash
maturin develop --features pytorch
```

## Structure

```
examples/mnist_pytorch/
├── README.md               # This file
├── model.py                # Shared CNN model
├── mnist_single.py         # Single-node baseline (no networking)
├── mnist_distributed_2.py   # 2-node distributed
└── mnist_distributed_3.py   # 3-node distributed
```

## Running

### Single Node (baseline)

```bash
cd examples/mnist_pytorch
python mnist_single.py
```

### 2 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2 python mnist_distributed_2.py

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1 python mnist_distributed_2.py
```

### 3 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2,3 python mnist_distributed_3.py

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1,3 python mnist_distributed_3.py

# Terminal 3
DF_NODE_ID=3 DF_PEERS=1,2 python mnist_distributed_3.py
```

## DeltaFabric API

```python
from delta_fabric import Fabric, Config

# 1. Create a fabric. This waits for the other nodes to be ready.
config = Config(peers=[2, 3])
fabric = Fabric(node_id=1, config=config)

# 2. Create PyTorch model (same seed on every node) and register its
#    initial weights before training. Otherwise the first sync sees weights
#    that already differ and triggers a one-time full-model resync.
model = MyModel()
model = fabric.step(model)

# 3. Training loop
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    # DeltaFabric sync - single call, returns updated model
    model = fabric.step(model)

# 4. Shutdown
fabric.close()
```

## Configuration

| Parameter               | Default | Description                     |
| ----------------------- | ------- | ------------------------------- |
| `gamma`                 | 0.02    | Consensus step size (keep <= 2 x `delta_selection_ratio`) |
| `delta_selection_ratio` | 0.01    | Only sync 1% of changed weights  |
| `sync_interval`         | 100     | Broadcast delta every N steps    |
| `value_encoding`        | f32     | Value format: `f32` (4 B), `int8` (1 B) or `sign` (1 bit) |
| `cluster`               | default | Job name; nodes only see nodes of the same cluster (examples read `DF_CLUSTER`) |

## Script Arguments

The example scripts support the following command-line arguments:

| Parameter    | Default | Description                                      |
| ------------ | ------- | ------------------------------------------------ |
| `--gamma`         | 0.02    | Overrides the default `gamma` in `Config` (0 disables syncing) |
| `--sync-interval` | 100     | Overrides the default `sync_interval` in `Config` |
| `--delta-selection-ratio` | 0.01    | Overrides the default `delta_selection_ratio` in `Config` |
| `--value-encoding` | f32     | Overrides the default `value_encoding` in `Config` |
| `--partition`     | contiguous | `class` gives each node only its own digits (3 nodes: 0-3, 4-6, 7-9), so sync quality shows in accuracy |
| `--use-data`      | 1.0     | Fraction of training data to use (0.0 to 1.0)     |

## Running Several Jobs on One Network

Nodes discover each other automatically, so two jobs using the same node IDs would mix. Give each job its own cluster name:

```bash
DF_CLUSTER=job-a DF_NODE_ID=1 DF_PEERS=2,3 python mnist_distributed_3.py
```

## Logging

DeltaFabric prints warnings (resyncs, rejected packets, an unstable `gamma`) to stderr by default. For more detail, call `delta_fabric.init_logging("info")` before creating the `Fabric`, or set `RUST_LOG`.

## Measuring Sync Quality

Each distributed node also writes `final_weights_<run>_node_<id>_gamma_<gamma>.pt` and `traffic_<run>_node_<id>_gamma_<gamma>.txt`, and prints how many bytes it sent. To see how close the nodes ended up, compare their final weights:

```bash
python ../compare_weights.py "final_weights_dist3_node_*_gamma_0.02.pt"
```

Run once with `--gamma 0` (no syncing) as a baseline: a working sync should give clearly smaller distances at a similar accuracy.

## Data Split

Training uses the full MNIST dataset (60,000 samples).

### Single Node

| Total Samples |
|---------------|
| 60,000 |

### 2 Nodes

Each node trains on 30,000 samples:
| Node | Samples |
|------|---------|
| 1 | 0 - 29,999 |
| 2 | 30,000 - 59,999 |

### 3 Nodes

Each node trains on 20,000 samples:
| Node | Samples |
|------|---------|
| 1 | 0 - 19,999 |
| 2 | 20,000 - 39,999 |
| 3 | 40,000 - 59,999 |
