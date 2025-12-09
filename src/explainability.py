import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from pathlib import Path


MODEL_PATH = "runs/detect/real_polyp5/weights/best.pt"
TEST_IMAGE = "sample_images/example_polyp.jpg"
SAVE_PATH = "experiments/results/gradcam/gradcam_result.png"

# Best Grad-CAM layer
TARGET_LAYER = "model.15.cv2"


def yolo_gradcam(model_path, image_path, save_path):
    print("🔥 Loading YOLOv8 model...")
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()

    # Load + preprocess image
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    transform = transforms.ToTensor()
    img_t = transform(img_resized).unsqueeze(0)
    img_t.requires_grad = True

    # REGISTER HOOKS
    activations, gradients = {}, {}

    target_layer = dict(model.named_modules())[TARGET_LAYER]

    def forward_hook(m, inp, out):
        activations["value"] = out

    def backward_hook(m, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # FORWARD
    print("🔥 Forward pass...")
    preds = model(img_t)

    # Highest objectness score
    score = preds[0][..., 4].max()
    score.backward()

    # BUILD CAM
    print("🔥 Building Grad-CAM...")

    A = activations["value"].detach()[0]        # (C, H, W)
    G = gradients["value"].detach()[0]          # (C, H, W)

    weights = G.mean(dim=(1, 2))                # (C,)

    cam = (A * weights[:, None, None]).sum(dim=0)   # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    # Normalize CAM
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    # ensure (H,W) uint8
    cam_uint8 = (cam * 255).astype("uint8")

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay
    result = (0.5 * heatmap + 0.5 * img_bgr).astype(np.uint8)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, result)

    print(f"🎉 Grad-CAM saved to: {save_path}")


if __name__ == "__main__":
    yolo_gradcam(MODEL_PATH, TEST_IMAGE, SAVE_PATH)
