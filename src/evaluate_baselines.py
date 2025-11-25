import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd

from src.models.yolo_baseline import YOLOv8Baseline
from src.models.maskrcnn_baseline import MaskRCNNBaseline, PolypDataset
from ultralytics import YOLO


class DetectionEvaluator:
    """
    Evaluator for object detection models with mAP calculation
    """
    
    def __init__(self, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        """
        Initialize evaluator
        
        Args:
            iou_thresholds: IoU thresholds for mAP calculation
        """
        self.iou_thresholds = iou_thresholds
        self.class_names = ['polyp']
        
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        # Intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    def calculate_map(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate mAP for a single IoU threshold
        
        Args:
            predictions: List of predictions per image
            targets: List of ground truth boxes per image
            iou_threshold: IoU threshold for TP/FP
        """
        all_detections = []
        all_ground_truths = []
        
        # Collect all detections and ground truths
        for img_idx, (pred_boxes, gt_boxes) in enumerate(zip(predictions, targets)):
            # Add ground truths
            for gt_box in gt_boxes:
                all_ground_truths.append({
                    'image_idx': img_idx,
                    'box': gt_box,
                    'detected': False
                })
            
            # Add predictions
            for pred_data in pred_boxes:
                if len(pred_data) == 5:
                    x1, y1, x2, y2, confidence = pred_data
                    pred_box = [x1, y1, x2, y2]
                else:
                    # Handle different formats if needed
                    pred_box = pred_data[:-1] if len(pred_data) > 4 else pred_data[:4]
                    confidence = pred_data[-1] if len(pred_data) > 4 else 1.0
                
                all_detections.append({
                    'image_idx': img_idx,
                    'box': pred_box,
                    'confidence': confidence,
                    'matched': False
                })
        
        # Sort detections by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        true_positives = 0
        false_positives = 0
        precisions = []
        recalls = []
        
        for detection in all_detections:
            img_idx = detection['image_idx']
            pred_box = detection['box']
            
            # Find matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(all_ground_truths):
                if gt['image_idx'] == img_idx and not gt['detected']:
                    iou = self.calculate_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if detection is correct
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                all_ground_truths[best_gt_idx]['detected'] = True
                detection['matched'] = True
            else:
                false_positives += 1
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using area under precision-recall curve
        if len(precisions) == 0:
            return 0.0
        
        # Add points for curve calculation
        precisions = [0] + precisions + [0]
        recalls = [0] + recalls + [1]
        
        # Calculate area using trapezoidal rule
        ap = 0
        for i in range(len(recalls) - 1):
            ap += (recalls[i+1] - recalls[i]) * (precisions[i] + precisions[i+1]) / 2
        
        return ap
        
    def evaluate_yolo_model(self, model, data_dir):
        """
        Evaluate YOLO model
        
        Args:
            model: Trained YOLO model
            data_dir: Dataset directory
        """
        print("📊 Evaluating YOLO model...")
        
        # Load validation dataset
        val_dir = Path(data_dir) / 'val' / 'images'
        val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        
        predictions = []
        targets = []
        
        for img_path in tqdm(val_images, desc="Processing images"):
            # Get ground truth (from YOLO label file)
            label_path = val_dir.parent / 'labels' / (img_path.stem + '.txt')
            gt_boxes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Convert YOLO format to absolute coordinates
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            
                            # Get image dimensions
                            img = Image.open(img_path)
                            img_w, img_h = img.size
                            
                            # Convert to absolute coordinates
                            x1 = int((x_center - width/2) * img_w)
                            y1 = int((y_center - height/2) * img_h)
                            x2 = int((x_center + width/2) * img_w)
                            y2 = int((y_center + height/2) * img_h)
                            
                            gt_boxes.append([x1, y1, x2, y2])
            
            # Get predictions - use very low confidence threshold for 5-shot models
            results = model.predict(str(img_path), conf=0.001)
            pred_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        pred_boxes.append([x1, y1, x2, y2, confidence])
            
            predictions.append(pred_boxes)
            targets.append(gt_boxes)
        
        # Calculate mAP at different IoU thresholds
        map_scores = {}
        for threshold in self.iou_thresholds:
            map_score = self.calculate_map(predictions, targets, threshold)
            map_scores[f'map@{threshold:.2f}'] = map_score
        
        # Extract commonly used metrics
        metrics = {
            'map50': map_scores['map@0.50'],
            'map50_95': np.mean(list(map_scores.values())),
            'map75': map_scores['map@0.75'],
            'num_images': len(val_images),
            'total_predictions': sum(len(pred) for pred in predictions),
            'total_ground_truths': sum(len(gt) for gt in targets)
        }
        
        return metrics, predictions, targets
        
    def evaluate_maskrcnn_model(self, model, dataset):
        """
        Evaluate Mask R-CNN model
        
        Args:
            model: Trained Mask R-CNN model
            dataset: Validation dataset
        """
        print("📊 Evaluating Mask R-CNN model...")
        
        predictions = []
        targets = []
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=model._collate_fn
        )
        
        model.model.eval()
        with torch.no_grad():
            for images, gt_targets in tqdm(data_loader, desc="Processing images"):
                image = images[0].to(model.device)
                gt_target = gt_targets[0]
                
                # Get ground truth boxes
                gt_boxes = gt_target['boxes'].cpu().numpy().tolist()
                targets.append(gt_boxes)
                
                # Get predictions - use very low confidence threshold for 5-shot models
                pred = model.predict(image, confidence=0.001)
                pred_boxes = []
                
                if len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box.tolist()
                        pred_boxes.append([x1, y1, x2, y2, score])
                
                predictions.append(pred_boxes)
        
        # Calculate mAP at different IoU thresholds
        map_scores = {}
        for threshold in self.iou_thresholds:
            map_score = self.calculate_map(predictions, targets, threshold)
            map_scores[f'map@{threshold:.2f}'] = map_score
        
        # Extract commonly used metrics
        metrics = {
            'map50': map_scores['map@0.50'],
            'map50_95': np.mean(list(map_scores.values())),
            'map75': map_scores['map@0.75'],
            'num_images': len(dataset),
            'total_predictions': sum(len(pred) for pred in predictions),
            'total_ground_truths': sum(len(gt) for gt in targets)
        }
        
        return metrics, predictions, targets
        
    def visualize_predictions(self, image_dir, predictions, targets, save_dir):
        """
        Visualize predictions vs ground truth
        
        Args:
            image_dir: Directory with images
            predictions: List of predictions
            targets: List of ground truth boxes
            save_dir: Directory to save visualizations
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        image_dir = Path(image_dir)
        images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        for i, img_path in enumerate(images[:10]):  # Visualize first 10 images
            if i >= len(predictions):
                break
                
            # Load image
            img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Draw ground truth (green)
            for gt_box in targets[i]:
                x1, y1, x2, y2 = gt_box
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                draw.text((x1, y1-15), 'GT', fill='green')
            
            # Draw predictions (red)
            for pred_box in predictions[i]:
                x1, y1, x2, y2, confidence = pred_box
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1-15), f'P: {confidence:.2f}', fill='red')
            
            # Save image
            img.save(save_dir / f'pred_{i:03d}.png')
        
        print(f"📊 Visualizations saved to {save_dir}")


def evaluate_all_models(args):
    """Evaluate all trained baseline models"""
    print("🎯 Evaluating baseline models...")
    
    evaluator = DetectionEvaluator()
    results = {}
    
    # Evaluate YOLO
    try:
        print("\n📊 Evaluating YOLOv8...")
        yolo = YOLOv8Baseline()
        yolo.load_model()
        
        # Load best trained model if available
        # Look for the most recent trained model
        runs_dir = Path('runs/detect')
        if runs_dir.exists():
            # Find the most recent yolov8n_5shot* directory
            model_dirs = [d for d in runs_dir.iterdir() if d.name.startswith('yolov8n_5shot') and d.is_dir()]
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_dir / 'weights' / 'best.pt'
                if best_model_path.exists():
                    print(f"Loading trained model from {best_model_path}")
                    yolo.model = YOLO(str(best_model_path))
        
        metrics, predictions, targets = evaluator.evaluate_yolo_model(yolo, args.data_dir)
        results['YOLOv8'] = metrics
        
        # Visualize some predictions
        evaluator.visualize_predictions(
            Path(args.data_dir) / 'val' / 'images',
            predictions, targets,
            'experiments/results/yolo_visualizations'
        )
        
    except Exception as e:
        print(f"❌ YOLO evaluation failed: {e}")
    
    # Evaluate Mask R-CNN
    try:
        print("\n📊 Evaluating Mask R-CNN...")
        maskrcnn = MaskRCNNBaseline(num_classes=2)
        maskrcnn.load_model()
        
        # Load best model if available (handle class mismatch gracefully)
        best_model_path = 'experiments/checkpoints/maskrcnn_5shot_best.pt'
        if Path(best_model_path).exists():
            try:
                maskrcnn.model.load_state_dict(torch.load(best_model_path, map_location=maskrcnn.device))
                print(f"✅ Loaded trained Mask R-CNN from {best_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load trained model (class mismatch): {e}")
                print("Using pretrained model instead")
        
        # Create validation dataset
        val_dataset = PolypDataset(
            root=Path(args.data_dir) / 'val' / 'images',
            transforms=maskrcnn.get_transform(train=False)
        )
        
        metrics, predictions, targets = evaluator.evaluate_maskrcnn_model(maskrcnn, val_dataset)
        results['MaskRCNN'] = metrics
        
        # Visualize some predictions
        evaluator.visualize_predictions(
            Path(args.data_dir) / 'val' / 'images',
            predictions, targets,
            'experiments/results/maskrcnn_visualizations'
        )
        
    except Exception as e:
        print(f"❌ Mask R-CNN evaluation failed: {e}")
    
    # Save results
    if results:
        with open('experiments/results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison table
        print("\n📈 Evaluation Results:")
        print("-" * 70)
        print(f"{'Model':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'mAP@0.75':<10} {'Images':<8}")
        print("-" * 70)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<12} {metrics['map50']:<10.4f} {metrics['map50_95']:<12.4f} "
                  f"{metrics['map75']:<10.4f} {metrics['num_images']:<8}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--data_dir', type=str, default='data/5shot',
                       help='Dataset directory')
    
    args = parser.parse_args()
    
    # Create results directory
    Path('experiments/results').mkdir(parents=True, exist_ok=True)
    
    # Evaluate models
    results = evaluate_all_models(args)
    
    print("\n🎉 Evaluation completed!")
    print("Results saved to experiments/results/evaluation_results.json")
