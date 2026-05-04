import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import Model

import time

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
TRAIN_SAMPLES = 60000

torch.manual_seed(SEED)


def main():
    parser = argparse.ArgumentParser(description="Single-node MNIST training")
    parser.add_argument(
        "--use-data",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0 to 1.0)",
    )
    args = parser.parse_args()

    print("Initializing single-node MNIST training")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    num_train_samples = int(TRAIN_SAMPLES * args.use_data)
    print(f"Using {num_train_samples} / {TRAIN_SAMPLES} training samples")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("Loading MNIST data...")
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    if args.use_data < 1.0:
        indices = list(range(num_train_samples))
        train_dataset = Subset(train_dataset, indices)

    test_dataset = datasets.MNIST(
        "./data", train=False, download=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model()
    print(f"Model initialized with {model.num_params()} parameters")

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")

    total_elapsed = 0
    epoch_times = []
    epoch_accuracies = []
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
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
        print(f"Epoch {epoch + 1}: Accuracy = {acc:.4f}, Time = {elapsed:.2f}s")

    print("Training complete")
    print(f"Final test accuracy: {acc:.4f}")
    print(f"Average time per epoch: {total_elapsed / EPOCHS:.2f}s")

    with open("epoch_times_single.txt", "w") as f:
        f.write(",".join(map(str, epoch_times)))

    with open("accuracy_single.txt", "w") as f:
        f.write(",".join(map(str, epoch_accuracies)))


if __name__ == "__main__":
    main()
