import torch
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path


class YOLOv8Baseline:
    """
    YOLOv8 baseline model for 5-shot images detection
    """
    
    def __init__(self, model_size='n', device='auto'):
        """
        Initialize YOLOv8 model
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_size = model_size
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['images']
        
    def load_model(self):
        """Load pretrained YOLOv8 model"""
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        print(f"✅ Loaded YOLOv8{self.model_size} on {self.device}")
        
    def prepare_dataset(self, data_dir, n_shot=5):
        """
        Prepare YOLO format dataset from 5-shot data
        
        Args:
            data_dir: Directory containing images images and annotations
            n_shot: Number of examples per class (5-shot)
        """
        # Create YOLO dataset structure
        yolo_data_dir = Path(data_dir) / 'yolo_format'
        yolo_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create real_dataset/val splits
        for split in ['real_dataset', 'val']:
            (yolo_data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        yaml_content = f"""
path: {yolo_data_dir.absolute()}
real_dataset: real_dataset/images
val: val/images

nc: 1
names: {self.class_names}
"""
        
        with open(yolo_data_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
            
        return yolo_data_dir / 'dataset.yaml'
        
    def train_5shot(self, data_yaml, epochs=100, imgsz=640, batch=4):
        """
        Train YOLOv8 on 5-shot data
        
        Args:
            data_yaml: Path to YOLO dataset.yaml
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
        """
        if self.model is None:
            self.load_model()
            
        print(f"🚀 Training YOLOv8{self.model_size} on 5-shot data...")
        
        # Train with small dataset settings
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=f'yolov8{self.model_size}_5shot',
            save=True,
            plots=True,
            device=self.device,
            # 5-shot specific settings
            lr0=0.001,  # Lower learning rate for small dataset
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=20,  # Early stopping
            save_period=10,
        )
        
        print("✅ Training completed!")
        return results
        
    def evaluate(self, data_yaml, imgsz=640):
        """
        Evaluate YOLOv8 model
        
        Args:
            data_yaml: Path to YOLO dataset.yaml
            imgsz: Image size
        """
        if self.model is None:
            self.load_model()
            
        print("📊 Evaluating YOLOv8 model...")
        
        results = self.model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            device=self.device,
            plots=True
        )
        
        # Extract metrics
        metrics = {
            'map50': results.box.map50,
            'map50_95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        }
        
        print(f"📈 Evaluation Results:")
        print(f"   mAP@0.5: {metrics['map50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        return metrics
        
    def predict(self, image_path, conf=0.25):
        """
        Make predictions on single image
        
        Args:
            image_path: Path to image
            conf: Confidence threshold
        """
        if self.model is None:
            self.load_model()
            
        results = self.model(image_path, conf=conf)
        return results
        
    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        self.model.save(path)
        print(f"✅ Model saved to {path}")


if __name__ == "__main__":
    # Example usage
    yolo = YOLOv8Baseline(model_size='n')
    yolo.load_model()
    
    # Prepare dataset (you need to implement this part with actual data)
    # data_yaml = yolo.prepare_dataset('data/5shot')
    
    # Train on 5-shot data
    # yolo.train_5shot(data_yaml)
    
    # Evaluate
    # metrics = yolo.evaluate(data_yaml)
    
    print("YOLOv8 baseline ready for 5-shot images detection!")
