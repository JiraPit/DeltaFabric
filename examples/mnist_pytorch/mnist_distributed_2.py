import os
import sys
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
TRAIN_SAMPLES_PER_NODE = TRAIN_SAMPLES // NUM_NODES

torch.manual_seed(SEED)


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def parse_peers(peers_str):
    if not peers_str:
        return []
    return [int(p) for p in peers_str.split(",") if p]


def main():
    node_id = int(os.environ["DF_NODE_ID"])
    peers_str = os.environ.get("DF_PEERS", "")
    peers = parse_peers(peers_str)

    partition_start = (node_id - 1) * TRAIN_SAMPLES_PER_NODE
    partition_end = partition_start + TRAIN_SAMPLES_PER_NODE

    print(f"Node {node_id}: Starting with peers {peers}")
    print(
        f"Node {node_id}: Partition {partition_start} - {partition_end} ({TRAIN_SAMPLES_PER_NODE} samples)"
    )

    global fabric
    config = Config(
        peers=peers, alpha=0.25, sync_interval=100, delta_selection_ratio=0.01
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
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        for _, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            old_weights = {k: v.clone() for k, v in model.state_dict().items()}
            model = fabric.step(model)

            # Calculate total number of changed elements
            changed_elements = 0
            for k in old_weights:
                changed_elements += torch.sum(
                    model.state_dict()[k] != old_weights[k]
                ).item()

            if changed_elements > 0:
                print(f"SYNC: Applied updates to {int(changed_elements)} parameters")

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
        elapsed = time.time() - start_time
        total_elapsed += elapsed
        print(
            f"Node {node_id}, Epoch {epoch + 1}: Accuracy = {acc:.4f}, Time = {elapsed:.2f}s"
        )

    print(f"Node {node_id}: Training complete")
    print(f"Node {node_id}: Average time per epoch: {total_elapsed / EPOCHS:.2f}s")

    fabric.close()
    print(f"Node {node_id}: Shutdown complete")


if __name__ == "__main__":
    main()
