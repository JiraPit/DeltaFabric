# DeltaFabric PyTorch Stanford Cars Example

Distributed Stanford Cars training demonstrating DeltaFabric's weight synchronization protocol with PyTorch.
This example uses a pre-trained ResNet50 model fine-tuned for 196 car classes.

## Prerequisites

Build and install the DeltaFabric Python extension:

```bash
cd ../../DeltaFabric
maturin develop --features pytorch
```

## Data Preparation

Use the provided script to download the dataset:

```bash
cd examples/stanford_cars_pytorch
./download_dataset.sh
```

## Structure

```
examples/stanford_cars_pytorch/
├── README.md               # This file
├── model.py                # ResNet50 model
├── stanford_cars_single.py         # Single-node baseline (no networking)
├── stanford_cars_distributed_2.py   # 2-node distributed
└── stanford_cars_distributed_3.py   # 3-node distributed
```

## Running

### Single Node (baseline)

```bash
cd examples/stanford_cars_pytorch
python stanford_cars_single.py
```

### 2 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2 python stanford_cars_distributed_2.py

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1 python stanford_cars_distributed_2.py
```

### 3 Nodes Distributed

```bash
# Terminal 1
DF_NODE_ID=1 DF_PEERS=2,3 python stanford_cars_distributed_3.py

# Terminal 2
DF_NODE_ID=2 DF_PEERS=1,3 python stanford_cars_distributed_3.py

# Terminal 3
DF_NODE_ID=3 DF_PEERS=1,2 python stanford_cars_distributed_3.py
```

## DeltaFabric API

```python
from delta_fabric import Fabric, Config

# 1. Create a fabric. This waits for other modules to be ready.
config = Config(peers=[2, 3])
fabric = Fabric(node_id=1, config=config)

# 2. Create PyTorch model
model = StanfordCarsModel()

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
| `alpha`                 | 0.1     | Blend factor for remote deltas    |
| `delta_selection_ratio` | 0.01    | Only sync 1% of changed weights  |
| `sync_interval`         | 100     | Broadcast delta every N steps    |
| `relay_threshold`       | 1e-6    | Minimum delta to relay           |

## Script Arguments

The example scripts support the following command-line arguments:

| Parameter    | Default | Description                                      |
| ------------ | ------- | ------------------------------------------------ |
| `--alpha`         | 0.25    | Overrides the default `alpha` in `Config`         |
| `--sync-interval` | 100     | Overrides the default `sync_interval` in `Config` |
| `--delta-selection-ratio` | 0.01    | Overrides the default `delta_selection_ratio` in `Config` |
| `--use-data`      | 1.0     | Fraction of training data to use (0.0 to 1.0)     |

## Data Split

Training uses the Stanford Cars dataset (8,144 training samples).

### Single Node

| Total Samples |
|---------------|
| 8,144 |

### 2 Nodes

Each node trains on 4,072 samples:
| Node | Samples |
|------|---------|
| 1 | 0 - 4,071 |
| 2 | 4,072 - 8,143 |

### 3 Nodes

Each node trains on 2,714 samples:
| Node | Samples |
|------|---------|
| 1 | 0 - 2,713 |
| 2 | 2,714 - 5,427 |
| 3 | 5,428 - 8,141 |
