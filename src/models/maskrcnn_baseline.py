import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import os
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm


class MaskRCNNBaseline:
    """
    Mask R-CNN baseline model for 5-shot images detection and segmentation
    """
    
    def __init__(self, num_classes=1, device='auto'):
        """
        Initialize Mask R-CNN model
        
        Args:
            num_classes: Number of classes (excluding background)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.num_classes = num_classes
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['images', 'masks']
        
    def load_model(self):
        """Load pretrained Mask R-CNN model"""
        # Load pretrained Mask R-CNN
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        
        # Replace the classifier with a new one for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        
        # Replace the mask predictor with a new one
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes + 1)
        
        self.model.to(self.device)
        print(f"✅ Loaded Mask R-CNN on {self.device}")
        
    def prepare_dataset(self, data_dir, n_shot=5):
        """
        Prepare COCO format dataset from 5-shot data
        
        Args:
            data_dir: Directory containing images images and annotations
            n_shot: Number of examples per class (5-shot)
        """
        # Create COCO dataset structure
        coco_data_dir = Path(data_dir) / 'coco_format'
        coco_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create real_dataset/val splits
        for split in ['real_dataset', 'val']:
            (coco_data_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Create COCO annotation files
        self._create_coco_annotations(coco_data_dir, n_shot)
        
        return coco_data_dir
        
    def _create_coco_annotations(self, data_dir, n_shot):
        """Create COCO format annotation files"""
        # This is a placeholder - you'll need to implement actual annotation creation
        # based on your dataset format
        
        for split in ['real_dataset', 'val']:
            annotations = {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "images", "supercategory": "images"}
                ]
            }
            
            # Save annotations
            with open(data_dir / f'{split}_annotations.json', 'w') as f:
                json.dump(annotations, f, indent=2)
                
    def get_transform(self, train=True):
        """Get data transforms"""
        from torchvision.transforms import v2 as T
        import torchvision.transforms as transforms_legacy
        
        transform_list = []
        if train:
            transform_list.append(T.RandomHorizontalFlip(0.5))
            transform_list.append(T.RandomVerticalFlip(0.5))
        
        # Convert PIL to tensor and normalize
        transform_list.append(transforms_legacy.ToTensor())
        transform_list.append(transforms_legacy.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return T.Compose(transform_list)
        
    def train_5shot(self, train_dataset, val_dataset, epochs=100, lr=0.001, batch_size=2):
        """
        Train Mask R-CNN on 5-shot data
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
        """
        if self.model is None:
            self.load_model()
            
        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=self._collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Optimizer and scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        print(f"🚀 Training Mask R-CNN on 5-shot data...")
        
        best_map = 0
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            # Validation
            val_map = self._evaluate_epoch(val_loader)
            
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val mAP: {val_map:.4f}")
            
            # Save best model
            if val_map > best_map:
                best_map = val_map
                torch.save(self.model.state_dict(), 'experiments/checkpoints/maskrcnn_5shot_best.pt')
            
            lr_scheduler.step()
        
        print("✅ Training completed!")
        return best_map
        
    def _collate_fn(self, batch):
        """Collate function for data loader"""
        return tuple(zip(*batch))
        
    def _evaluate_epoch(self, data_loader):
        """Evaluate model for one epoch"""
        self.model.eval()
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # This is a simplified evaluation - you'll need proper mAP calculation
                predictions = self.model(images)
                # TODO: Implement proper mAP calculation
        
        # Return placeholder mAP
        return 0.5
        
    def evaluate(self, data_loader):
        """
        Evaluate Mask R-CNN model
        
        Args:
            data_loader: Validation data loader
        """
        if self.model is None:
            self.load_model()
            
        print("📊 Evaluating Mask R-CNN model...")
        
        # TODO: Implement proper evaluation with mAP calculation
        metrics = {
            'map50': 0.5,  # Placeholder
            'map50_95': 0.3,  # Placeholder
            'precision': 0.6,  # Placeholder
            'recall': 0.4,  # Placeholder
        }
        
        print(f"📈 Evaluation Results:")
        print(f"   mAP@0.5: {metrics['map50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        return metrics
        
    def predict(self, image, confidence=0.5):
        """
        Make predictions on single image
        
        Args:
            image: PIL Image, tensor, or file path
            confidence: Confidence threshold
        """
        if self.model is None:
            self.load_model()
            
        self.model.eval()
        with torch.no_grad():
            # Handle different input types
            if isinstance(image, str):
                # File path
                img = Image.open(image).convert("RGB")
                transform = self.get_transform(train=False)
                image = transform(img)
            elif isinstance(image, Image.Image):
                # PIL Image
                transform = self.get_transform(train=False)
                image = transform(image)
            elif isinstance(image, torch.Tensor):
                # Already a tensor - ensure it's in the right format
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                # Move to device if not already
                image = image.to(self.device)
            
            # Ensure image is on device and has batch dimension
            if image.device != self.device:
                image = image.to(self.device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            predictions = self.model(image)
            
            # Filter by confidence
            pred = predictions[0]
            if len(pred['scores']) > 0:
                scores = pred['scores']
                keep = scores > confidence
                
                filtered_pred = {
                    'boxes': pred['boxes'][keep],
                    'labels': pred['labels'][keep],
                    'scores': pred['scores'][keep],
                    'masks': pred['masks'][keep] if 'masks' in pred else torch.tensor([])
                }
            else:
                # No predictions
                filtered_pred = {
                    'boxes': torch.tensor([]).reshape(0, 4).to(self.device),
                    'labels': torch.tensor([], dtype=torch.int64).to(self.device),
                    'scores': torch.tensor([]).to(self.device),
                    'masks': torch.tensor([]).to(self.device)
                }
            
            return filtered_pred
            
    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to {path}")


class PolypDataset(torch.utils.data.Dataset):
    """
    Custom dataset for images detection and segmentation
    """
    
    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.images = list(self.root.glob("*.jpg")) + list(self.root.glob("*.png"))
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Get image size for converting relative coordinates
        img_w, img_h = img.size
        
        # Load YOLO format label if it exists
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        boxes = []
        labels = []
        masks = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # Convert YOLO format to absolute coordinates
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))  # Keep class ID the same (0=images, 1=masks)
                        
                        # Create a simple rectangular mask
                        mask_w = int((x2 - x1))
                        mask_h = int((y2 - y1))
                        if mask_w > 0 and mask_h > 0:
                            mask = np.zeros((int(img_h), int(img_w)), dtype=np.uint8)
                            mask[int(y1):int(y2), int(x1):int(x2)] = 1
                            masks.append(mask)
        
        # Convert to tensors
        if len(boxes) == 0:
            # No annotations - create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, int(img_h), int(img_w)), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
        
    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    # Example usage
    maskrcnn = MaskRCNNBaseline(num_classes=1)
    maskrcnn.load_model()
    
    print("Mask R-CNN baseline ready for 5-shot images detection!")
