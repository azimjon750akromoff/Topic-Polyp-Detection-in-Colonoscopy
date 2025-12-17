"""Simple Colonoscopy Polyp Detection GUI

Uses two models:
1. YOLOv8 – object detection
2. Few-shot Mask R-CNN – detection via bounding boxes

Select the model in the radio button and press Run.
"""
import sys
from pathlib import Path
import time

import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import ops
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Path setup – make sure we can import from src.* when executing this file. The
# file lives in src/, so the project root is one directory above.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.maskrcnn_baseline import MaskRCNNBaseline  # pylint: disable=wrong-import-position

# -----------------------------------------------------------------------------
# Device and model loading
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv8 model (weights trained by user)
YOLO_WEIGHTS = "src/polyp_detection/full_real_yolov8n/weights/best.pt"
yolo_detector = YOLO(YOLO_WEIGHTS)

# YOLOv8-seg model for segmentation (alternative to MaskRCNN)
YOLO_SEG_WEIGHTS = "yolov8n-seg.pt"
yolo_seg_detector = YOLO(YOLO_SEG_WEIGHTS)

# Standard Mask R-CNN model (from original gui.py)
mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
mask_model.to(DEVICE).eval()
transform = torchvision.transforms.ToTensor()


# -----------------------------------------------------------------------------
# Detection helper
# -----------------------------------------------------------------------------

def detect(image: Image.Image, model_name: str, conf: float, iou_thr: float, min_area: int):
    """Run detection using the selected model and return annotated image + stats."""
    if image is None:
        return None, {"error": "No image uploaded"}, "❌ No image uploaded"

    start = time.time()
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    count = 0

    # Model inference ---------------------------------------------------------

    if model_name == "YOLOv8":
        results = yolo_detector.predict(
            source=np.array(image),
            conf=conf,
            iou=iou_thr,
            imgsz=640,
            agnostic_nms=True,
            max_det=20,
            verbose=False,
        )
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline="#22c55e", width=3)
                count += 1

    elif model_name == "YOLOv8-Segmentation":
        print(f"DEBUG: Running YOLOv8-Segmentation with confidence={conf}")
        results = yolo_seg_detector.predict(
            source=np.array(image),
            conf=conf,
            iou=iou_thr,
            imgsz=640,
            agnostic_nms=True,
            max_det=20,
            verbose=False,
        )
        print(f"DEBUG: YOLOv8-Segmentation results: {len(results)} result(s)")
        for i, r in enumerate(results):
            print(f"DEBUG: Result {i}: {len(r.boxes)} boxes")
            if len(r.boxes) > 0:
                scores = [b.conf.item() for b in r.boxes]
                print(f"DEBUG: Scores: {scores}")
        
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    print(f"DEBUG: Skipping box - area {area} < min_area {min_area}")
                    continue
                draw.rectangle([x1, y1, x2, y2], outline="#f59e0b", width=3)
                count += 1

    elif model_name == "MaskRCNN":
        print(f"DEBUG: Running standard MaskRCNN with confidence={conf}")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = mask_model(tensor)[0]

        # Strong filter (like in original)
        keep = pred["scores"] > 0.5
        boxes = pred["boxes"][keep]
        scores = pred["scores"][keep]

        print(f"DEBUG: MaskRCNN raw - {len(pred['boxes'])} boxes, after filter - {len(boxes)} boxes")
        if len(scores) > 0:
            score_values = [s.item() for s in scores]
            print(f"DEBUG: Filtered scores: {score_values}")

        if len(boxes) > 0:
            idx = ops.nms(boxes, scores, iou_thr)[:3]  # max 3
            boxes = boxes[idx]
            print(f"DEBUG: After NMS - {len(boxes)} boxes")

        for b in boxes:
            x1, y1, x2, y2 = map(int, b.cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                print(f"DEBUG: Skipping box - area {area} < min_area {min_area}")
                continue
            print(f"DEBUG: Drawing MaskRCNN box at ({x1},{y1}) to ({x2},{y2})")
            draw.rectangle([x1, y1, x2, y2], outline="#38bdf8", width=4)
            count += 1
    else:
        return None, {"error": "Unknown model"}, f"❌ Unknown model: {model_name}"

    # -------------------------------------------------------------------------
    stats = {
        "model": model_name,
        "polyps": count,
        "time_sec": round(time.time() - start, 2),
    }
    summary_md = f"""
### Detection Summary
- **Model:** {model_name}
- **Polyps detected:** {count}
- **Inference time:** {stats['time_sec']} s
"""
    return annotated, stats, summary_md

# -----------------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Colonoscopy Polyp Detection") as demo:
        gr.Markdown("""
        # 🩺 Colonoscopy Polyp Detection
        Select a model and upload an image to detect polyps.
        """)

        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(type="pil", label="Input image")
                model_radio = gr.Radio(
                    ["YOLOv8", "YOLOv8-Segmentation", "MaskRCNN"], 
                    value="YOLOv8", 
                    label="Detection Model",
                    info="Choose: YOLOv8 (trained), YOLOv8-Segmentation (pre-trained), or MaskRCNN (your model)"
                )
                conf_slider = gr.Slider(0.05, 0.9, 0.2, step=0.05, label="Confidence")
                iou_slider = gr.Slider(0.1, 0.7, 0.4, step=0.05, label="IoU threshold")
                min_area_slider = gr.Slider(100, 3000, 400, step=50, label="Min bounding-box area (px²)")
                run_btn = gr.Button("Run detection 🚀")

            with gr.Column():
                out_img = gr.Image(label="Annotated output")
                stats_json = gr.JSON(label="Statistics")
                summary_md = gr.Markdown()

        run_btn.click(
            detect,
            inputs=[inp_img, model_radio, conf_slider, iou_slider, min_area_slider],
            outputs=[out_img, stats_json, summary_md],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
