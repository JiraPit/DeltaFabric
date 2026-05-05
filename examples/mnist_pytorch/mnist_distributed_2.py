import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from model import Model

import time
from delta_fabric import Fabric, Config

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
NUM_NODES = 2
TRAIN_SAMPLES = 60000

torch.manual_seed(SEED)


# Wraps a dataset to index into specific indices (for data partitioning).
# [NOTE] This is example helper function, not part of DeltaFabric usage.
class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


# Parse comma-separated peer node IDs from environment variable.
# [NOTE] This is example helper function, not part of DeltaFabric usage.
def parse_peers(peers_str):
    if not peers_str:
        return []
    return [int(p) for p in peers_str.split(",") if p]


def main():
    parser = argparse.ArgumentParser(
        description="Distributed MNIST training with DeltaFabric"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Alpha value for DeltaFabric config"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=100,
        help="Steps between DeltaFabric syncs",
    )
    parser.add_argument(
        "--delta-selection-ratio",
        type=float,
        default=0.01,
        help="Delta selection ratio for DeltaFabric config",
    )
    parser.add_argument(
        "--use-data",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0 to 1.0)",
    )
    args = parser.parse_args()

    node_id = int(os.environ["DF_NODE_ID"])
    peers_str = os.environ.get("DF_PEERS", "")
    peers = parse_peers(peers_str)

    actual_train_samples = int(TRAIN_SAMPLES * args.use_data)
    train_samples_per_node = actual_train_samples // NUM_NODES

    partition_start = (node_id - 1) * train_samples_per_node
    partition_end = partition_start + train_samples_per_node

    print(f"Node {node_id}: Starting with peers {peers}")
    print(
        f"Node {node_id}: Partition {partition_start} - {partition_end} ({train_samples_per_node} samples)"
    )

    global fabric
    config = Config(
        peers=peers,
        alpha=args.alpha,
        sync_interval=args.sync_interval,
        delta_selection_ratio=args.delta_selection_ratio,
    )
    fabric = Fabric(node_id=node_id, config=config)

    print(f"Node {node_id}: DeltaFabric initialized")

    model = Model()
    print(f"Node {node_id}: Model initialized with {model.num_params()} parameters")

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("Loading MNIST data...")
    full_train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=False, transform=transform
    )
    print("Starting training with DeltaFabric sync...")

    my_indices = list(range(partition_start, partition_end))
    node_dataset = IndexedDataset(full_train_dataset, my_indices)
    train_loader = DataLoader(node_dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_elapsed = 0
    epoch_times = []
    epoch_accuracies = []
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        for _, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            {k: v.clone() for k, v in model.state_dict().items()}
            model = fabric.step(model)

        model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        with torch.no_grad():
            for images, targets in test_loader:
                output = model(images)
                predictions = output.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        epoch_accuracies.append(acc)
        elapsed = time.time() - start_time
        total_elapsed += elapsed
        epoch_times.append(elapsed)
        print(
            f"Node {node_id}, Epoch {epoch + 1}: Accuracy = {acc:.4f}, Time = {elapsed:.2f}s"
        )

    print(f"Node {node_id}: Training complete")
    print(f"Node {node_id}: Average time per epoch: {total_elapsed / EPOCHS:.2f}s")

    with open(f"epoch_times_dist2_node_{node_id}.txt", "w") as f:
        f.write(",".join(map(str, epoch_times)))

    with open(f"accuracy_dist2_node_{node_id}_alpha_{config.alpha}.txt", "w") as f:
        f.write(",".join(map(str, epoch_accuracies)))

    fabric.close()
    print(f"Node {node_id}: Shutdown complete")


if __name__ == "__main__":
    main()
