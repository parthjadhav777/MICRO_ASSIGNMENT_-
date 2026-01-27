import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. Dataset (Same as before) ---
class BehaviorDataset(Dataset):
    def __init__(self, num_samples=500, grid_size=32):
        self.inputs = []
        self.targets = []
        moves = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        for _ in range(num_samples):
            path = [[grid_size//2, grid_size//2]]
            curr_dir = np.random.choice([0, 1, 2, 3])
            length = np.random.randint(15, 25)
            
            input_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            behavior_map = np.zeros((grid_size, grid_size), dtype=np.longlong)
            
            for _ in range(length):
                x, y = path[-1]
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    input_grid[y, x] = 1.0
                
                action = np.random.choice(['straight', 'turn'], p=[0.6, 0.4])
                if action == 'turn':
                    curr_dir = (curr_dir + np.random.choice([-1, 1])) % 4
                    label = 2
                else:
                    label = 1
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    behavior_map[y, x] = label
                
                dx, dy = moves[curr_dir]
                path.append([x+dx, y+dy])

            self.inputs.append(torch.tensor(input_grid).unsqueeze(0))
            self.targets.append(torch.tensor(behavior_map))

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# --- 2. Model (Same U-Net) ---
class BehaviorNet(nn.Module):
    def __init__(self):
        super(BehaviorNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(),
            nn.Conv2d(8, 3, 1) # Output 3 Classes
        )

    def forward(self, x):
        return self.dec(self.enc(x))

# --- 3. Run with Enhanced Logging ---
if __name__ == "__main__":
    dataset = BehaviorDataset(num_samples=800) # Increased data slightly
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = BehaviorNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':<6} | {'Loss':<8} | {'Accuracy':<8}")
    print("-" * 30)

    model.train()
    for epoch in range(30):
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x) # Shape: (Batch, 3, 32, 32)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # --- CALCULATE ACCURACY ---
            # Get the predicted class index (0, 1, or 2)
            preds = torch.argmax(logits, dim=1) 
            
            # Compare with ground truth (y)
            correct_pixels = (preds == y).sum().item()
            total_correct += correct_pixels
            total_pixels += y.numel() # Total pixels in batch

        avg_loss = total_loss / len(loader)
        accuracy = (total_correct / total_pixels) * 100
        print(f"{epoch+1:<6} | {avg_loss:.4f}   | {accuracy:.2f}%")

    # --- 4. Enhanced Visualization ---
    model.eval()
    inp, targ = dataset[0]
    with torch.no_grad():
        logits = model(inp.unsqueeze(0))
        pred_map = torch.argmax(logits, dim=1).squeeze(0)

    # Setup Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Input
    ax[0].imshow(inp[0], cmap='gray_r') # Inverted gray for better visibility
    ax[0].set_title("Input (Trajectory)")
    ax[0].axis('off')

    # Define color map: 0=Purple (Bg), 1=Teal (Straight), 2=Yellow (Turn)
    cmap = plt.cm.viridis
    
    # 2. Ground Truth
    im1 = ax[1].imshow(targ, cmap=cmap, vmin=0, vmax=2)
    ax[1].set_title("Ground Truth (Labels)")
    ax[1].axis('off')

    # 3. Prediction
    im2 = ax[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=2)
    ax[2].set_title("Model Prediction")
    ax[2].axis('off')

    # Add Legend
    values = [0, 1, 2]
    labels = ["Background (0)", "Straight (1)", "Turn (2)"]
    colors = [cmap(v/2.0) for v in values]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.95, 0.85))

    plt.tight_layout()
    plt.show()