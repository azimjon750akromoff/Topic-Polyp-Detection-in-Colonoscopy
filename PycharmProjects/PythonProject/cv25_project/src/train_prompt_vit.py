import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from datetime import datetime
from src.models.prompt_vit import PromptTunedViT


class DummyPolypDataset(Dataset):
    def __init__(self, num_samples=50, img_size=224, num_classes=2):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(3, self.img_size, self.img_size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def train_model(
    num_epochs=3,
    batch_size=4,
    lr=5e-4,
    prompt_len=10,
    num_classes=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\n🚀 Training started on [{device}]...\n")

    train_dataset = DummyPolypDataset(num_samples=100, num_classes=num_classes)
    val_dataset = DummyPolypDataset(num_samples=20, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PromptTunedViT(num_classes=num_classes, prompt_len=prompt_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.head.parameters()) + [model.prompts],
        lr=lr,
        weight_decay=0.01,
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {epoch_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"📊 Validation Accuracy: {acc:.2f}%\n")

    os.makedirs("experiments/prompt_vit", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"experiments/prompt_vit/prompt_vit_{timestamp}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")


if __name__ == "__main__":
    train_model(num_epochs=3, batch_size=4, lr=5e-4)
