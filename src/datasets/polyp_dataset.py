import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PolypDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.images = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for label in ["images", "masks"]:
            folder = os.path.join(root_dir, label)
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")):
                    self.images.append(os.path.join(folder, f))
                    self.labels.append(1 if label == "images" else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
