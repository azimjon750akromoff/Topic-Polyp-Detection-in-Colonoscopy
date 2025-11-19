import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.prompt_vit import PromptTunedViT


def generate_cam(model, x):
    features = None

    def hook(module, inp, out):
        nonlocal features
        features = out

    h = model.base_model.blocks[-1].register_forward_hook(hook)

    outputs = model(x)
    pred = outputs.argmax(1).item()

    model.zero_grad()
    loss = outputs[0, pred]
    loss.backward()

    feats = features[0]
    cams = feats.mean(-1).detach().cpu().numpy()

    side = int(np.sqrt(len(cams) - 1))
    cam = cams[1:1 + side*side].reshape(side, side)
    cam = np.maximum(cam, 0)
    cam /= cam.max()

    h.remove()
    return cam, pred


def explain(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PromptTunedViT()
    model.load_state_dict(torch.load("experiments/checkpoints/prompt_vit_real.pt", map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    cam, pred = generate_cam(model, x)

    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_img = cam_img.resize(img.size)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(img)
    plt.imshow(cam_img, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    explain("sample_images/example_polyp.jpg")
