import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.prompt_vit import PromptTunedViT
from src.datasets.polyp_dataset import PolypDataset



def train_model(epochs=10, batch=8, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Training on:", device)

    model = PromptTunedViT().to(device)
    train_ds = PolypDataset("data/real_dataset")
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        loss_total = 0

        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f"Epoch {epoch+1} | Loss = {loss_total / len(train_dl):.4f}")

    torch.save(model.state_dict(), "experiments/checkpoints/prompt_vit_real.pt")
    print("Model saved!")


if __name__ == "__main__":
    train_model()
