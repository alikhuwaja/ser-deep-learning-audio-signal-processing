# src/train.py
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# ----- tiny config -----
N_MELS = 64
N_CLASSES = 6
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----- dummy dataset -----
class DummyAudioDataset(Dataset):
    def __init__(self, n_samples=64, n_mels=N_MELS, t=100, n_classes=N_CLASSES):
        self.x = torch.randn(n_samples, 1, n_mels, t)
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ----- simple model (CRNN-ish) -----
class SimpleCRNN(torch.nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )
        self.gru = torch.nn.GRU(
            input_size=32 * (n_mels // 4),
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(64 * 2, n_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        z = self.cnn(x)               # (B, C, F, T')
        B, C, F, T = z.shape
        z = z.permute(0, 3, 1, 2)     # (B, T', C, F)
        z = z.reshape(B, T, C * F)    # (B, T', feat)
        out, _ = self.gru(z)          # (B, T', 2*hidden)
        out = out[:, -1, :]           # last time step
        logits = self.fc(out)         # (B, n_classes)
        return logits

# ----- training loop -----
def train_epoch(model, loader, optim, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = DummyAudioDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCRNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optim, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss={loss:.4f}")

if __name__ == "__main__":
    main()
