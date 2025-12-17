import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.models.prompt_vit import PromptTunedViT
from src.datasets.polyp_dataset import PolypDataset


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Evaluating on:", device)

    model = PromptTunedViT()
    model.load_state_dict(torch.load("experiments/checkpoints/prompt_vit_real.pt", map_location=device))
    model.to(device).eval()

    val_ds = PolypDataset("data/val")
    val_dl = DataLoader(val_ds, batch_size=4)

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
