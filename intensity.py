import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np

import datashader as ds
import matplotlib.pyplot as plt
from colorcet import fire

# ----------------------------------------------------
# 1. Dataset: Trajectories → Datashader Raster
# ----------------------------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=300, grid_size=64):
        self.data = []
        self.labels = []
        self.canvas = ds.Canvas(
            plot_width=grid_size,
            plot_height=grid_size
        )

        print("Generating trajectories and rasterizing...")

        for _ in range(num_samples):
            # A. Random walk trajectory
            steps = np.random.randn(100, 2)
            path = np.cumsum(steps, axis=0)
            df = pd.DataFrame(path, columns=["x", "y"])

            # B. Rasterize trajectory
            agg = self.canvas.points(df, "x", "y")
            img = agg.values
            img = np.nan_to_num(img)

            if img.max() > 0:
                img = img / img.max()

            img_tensor = torch.tensor(
                img, dtype=torch.float32
            ).unsqueeze(0)  # (1, H, W)

            self.data.append(img_tensor)

            # C. Pseudo label: spread vs compact
            label = 1 if np.var(path) > 10 else 0
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ----------------------------------------------------
# 2. Tiny CNN
# ----------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(16 * 16 * 16, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ----------------------------------------------------
# 3. Accuracy computation
# ----------------------------------------------------
def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total


# ----------------------------------------------------
# 4. Visualization
# ----------------------------------------------------
def visualize_predictions(model, dataset, num_samples=6):
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_samples):
            img, label = dataset[i]
            output = model(img.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()

            axes[i].imshow(img.squeeze(0), cmap="inferno")
            axes[i].set_title(f"GT: {label} | Pred: {pred}")
            axes[i].axis("off")

    plt.suptitle("Trajectory Heatmaps — Ground Truth vs Prediction")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 5. Training Pipeline
# ----------------------------------------------------
def train_pipeline():
    dataset = TrajectoryDataset(num_samples=300)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = TinyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting training...\n")

    for epoch in range(20):
        total_loss = 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    print("\nTraining complete.")
    visualize_predictions(model, dataset)


# ----------------------------------------------------
# 6. Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    train_pipeline()
