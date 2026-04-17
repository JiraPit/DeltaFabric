# DeltaFabric PyTorch MNIST Example

Distributed MNIST training demonstrating DeltaFabric's weight synchronization protocol with PyTorch.

## Prerequisites

Build and install the DeltaFabric Python extension:

```bash
cd ../../DeltaFabric
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

# 1. Create config with sensible defaults
config = Config(peers=[2, 3])
fabric = Fabric(node_id=1, config=config)

# 2. Create PyTorch model
model = MyModel()

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
| `alpha`                 | 0.5     | Blend factor for remote deltas    |
| `delta_selection_ratio` | 0.01    | Only sync 1% of changed weights  |
| `sync_interval`         | 100     | Broadcast delta every N steps    |
| `relay_threshold`       | 1e-6    | Minimum delta to relay           |

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
