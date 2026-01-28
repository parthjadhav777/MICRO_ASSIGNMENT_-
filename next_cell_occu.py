# ==========================================
# FILE: cnn_next_cell_logged.py
# Task: Predict Next Cell Occupancy (Regression)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Dataset
# -------------------------------
class NextCellDataset(Dataset):
    def __init__(self, num_samples=500, grid_size=32):
        self.inputs, self.targets = [], []
        self.grid_size = grid_size

        moves = [(0,1),(1,0),(0,-1),(-1,0)]

        for _ in range(num_samples):
            path = [[grid_size//2, grid_size//2]]
            length = np.random.randint(10, 20)

            input_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

            for _ in range(length):
                x, y = path[-1]
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    input_grid[y, x] = 1.0
                dx, dy = moves[np.random.randint(4)]
                path.append([x+dx, y+dy])

            target_grid = np.zeros_like(input_grid)
            last_x, last_y = path[-1]
            if 0 <= last_x < grid_size and 0 <= last_y < grid_size:
                target_grid[last_y, last_x] = 1.0

            self.inputs.append(torch.tensor(input_grid).unsqueeze(0))
            self.targets.append(torch.tensor(target_grid).unsqueeze(0))

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# -------------------------------
# 2. Model
# -------------------------------
class NextCellNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,2,stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16,8,2,stride=2), nn.ReLU(),
            nn.Conv2d(8,1,1)
        )

    def forward(self, x):
        return self.dec(self.enc(x))

# -------------------------------
# 3. Utility Functions
# -------------------------------
def argmax_2d(tensor):
    """Returns (x, y) of max value"""
    flat_idx = torch.argmax(tensor)
    y = flat_idx // tensor.shape[-1]
    x = flat_idx % tensor.shape[-1]
    return int(x), int(y)

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# -------------------------------
# 4. Training
# -------------------------------
if __name__ == "__main__":

    dataset = NextCellDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NextCellNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    history = {
        "loss": [],
        "peak_value": [],
        "distance_error": []
    }

    print("\nTraining Next-Cell Predictor\n")

    for epoch in range(30):
        model.train()
        total_loss = 0

        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history["loss"].append(avg_loss)

        # --- Quick Evaluation Sample ---
        model.eval()
        with torch.no_grad():
            inp, targ = dataset[np.random.randint(len(dataset))]
            pred = model(inp.unsqueeze(0))[0,0]

            pred_xy = argmax_2d(pred)
            true_xy = argmax_2d(targ[0])

            dist = euclidean(pred_xy, true_xy)
            peak = pred.max().item()

        history["distance_error"].append(dist)
        history["peak_value"].append(peak)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {avg_loss:.5f} | "
            f"Peak: {peak:.3f} | "
            f"Dist Err: {dist:.2f}"
        )

    # -------------------------------
    # 5. Visualization
    # -------------------------------
    model.eval()
    inp, targ = dataset[0]
    with torch.no_grad():
        pred = model(inp.unsqueeze(0))[0,0]

    pred_xy = argmax_2d(pred)
    true_xy = argmax_2d(targ[0])

    fig, ax = plt.subplots(1, 4, figsize=(16,4))

    ax[0].imshow(inp[0], cmap='gray')
    ax[0].set_title("Input Trajectory")

    ax[1].imshow(targ[0], cmap='jet')
    ax[1].scatter(*true_xy, c='white', s=50)
    ax[1].set_title("True Next Cell")

    ax[2].imshow(pred, cmap='jet')
    ax[2].scatter(*pred_xy, c='white', s=50)
    ax[2].set_title("Predicted Heatmap")

    ax[3].plot(history["loss"], label="Loss")
    ax[3].plot(history["distance_error"], label="Dist Error")
    ax[3].legend()
    ax[3].set_title("Training Metrics")

    plt.show()
