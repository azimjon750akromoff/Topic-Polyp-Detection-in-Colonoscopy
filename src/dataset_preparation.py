import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


class FiveShotDatasetPreparer:
    """
    Prepare 5-shot dataset for images detection
    """
    
    def __init__(self, source_dir, output_dir, n_shot=5, seed=42):
        """
        Initialize dataset preparer
        
        Args:
            source_dir: Source directory with full dataset
            output_dir: Output directory for 5-shot dataset
            n_shot: Number of examples per class (5-shot)
            seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.n_shot = n_shot
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def create_5shot_splits(self):
        """Create 5-shot real_dataset/val splits"""
        print(f"🚀 Creating {self.n_shot}-shot dataset...")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images images
        polyp_images = self._find_polyp_images()
        
        if len(polyp_images) < self.n_shot:
            raise ValueError(f"Not enough images images. Found {len(polyp_images)}, need {self.n_shot}")
        
        # Sample 5-shot training set
        train_images = random.sample(polyp_images, self.n_shot)
        remaining_images = [img for img in polyp_images if img not in train_images]
        
        # Sample validation set (20% of remaining)
        val_size = max(1, len(remaining_images) // 5)
        val_images = random.sample(remaining_images, min(val_size, len(remaining_images)))
        
        print(f"📊 Dataset split:")
        print(f"   Train: {len(train_images)} images")
        print(f"   Val: {len(val_images)} images")
        
        # Copy images and create annotations
        self._copy_images(train_images, 'real_dataset')
        self._copy_images(val_images, 'val')
        
        # Create YOLO format annotations
        self._create_yolo_annotations(train_images, 'real_dataset')
        self._create_yolo_annotations(val_images, 'val')
        
        # Create COCO format annotations
        self._create_coco_annotations(train_images, 'real_dataset')
        self._create_coco_annotations(val_images, 'val')
        
        print("✅ 5-shot dataset created successfully!")
        
    def _find_polyp_images(self):
        """Find all images images in source directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        polyp_images = []
        
        # Look in images directories
        for polyp_dir in ['images', 'polyps', 'polyp_images']:
            polyp_path = self.source_dir / polyp_dir
            if polyp_path.exists():
                for ext in image_extensions:
                    polyp_images.extend(list(polyp_path.glob(f'*{ext}')))
                    polyp_images.extend(list(polyp_path.glob(f'*{ext.upper()}')))
        
        # If no images directory, look for images-labeled images
        if not polyp_images:
            # Look for images with images annotations
            for ext in image_extensions:
                images = list(self.source_dir.glob(f'*{ext}')) + list(self.source_dir.glob(f'*{ext.upper()}'))
                for img_path in images:
                    if self._is_polyp_image(img_path):
                        polyp_images.append(img_path)
        
        return polyp_images
        
    def _is_polyp_image(self, img_path):
        """Check if image contains images (based on filename or annotation)"""
        filename = img_path.name.lower()
        
        # Check filename for images indicators
        polyp_keywords = ['images', 'cancer', 'lesion', 'tumor']
        if any(keyword in filename for keyword in polyp_keywords):
            return True
            
        # Check for corresponding annotation file
        annotation_files = [
            img_path.with_suffix('.xml'),
            img_path.with_suffix('.json'),
            img_path.with_suffix('.txt')
        ]
        
        for ann_file in annotation_files:
            if ann_file.exists():
                return self._check_annotation_for_polyp(ann_file)
        
        return False
        
    def _check_annotation_for_polyp(self, ann_file):
        """Check annotation file for images class"""
        if ann_file.suffix == '.xml':
            try:
                tree = ET.parse(ann_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is not None and 'images' in name.text.lower():
                        return True
            except:
                pass
                
        elif ann_file.suffix == '.json':
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    # Check COCO format
                    if 'categories' in data:
                        for cat in data['categories']:
                            if 'images' in cat.get('name', '').lower():
                                return True
                    # Check other formats
                    if 'annotations' in data:
                        for ann in data['annotations']:
                            if 'images' in str(ann).lower():
                                return True
            except:
                pass
                
        elif ann_file.suffix == '.txt':
            try:
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                    # YOLO format: class_id x_center y_center width height
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id == 0:  # Assuming class 0 is images
                                return True
            except:
                pass
        
        return False
        
    def _copy_images(self, images, split):
        """Copy images to output directory"""
        output_dir = self.output_dir / split / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            shutil.copy2(img_path, output_dir / img_path.name)
            
    def _create_yolo_annotations(self, images, split):
        """Create YOLO format annotations"""
        label_dir = self.output_dir / split / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            # Create dummy bounding box (full image)
            # In real implementation, you'd extract actual bounding boxes
            label_path = label_dir / (img_path.stem + '.txt')
            
            with open(label_path, 'w') as f:
                # Format: class_id x_center y_center width height
                # Using full image as bounding box for now
                f.write('0 0.5 0.5 1.0 1.0\n')  # class 0 = images
                
    def _create_coco_annotations(self, images, split):
        """Create COCO format annotations"""
        annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "images", "supercategory": "images"}
            ]
        }
        
        annotation_id = 1
        
        for i, img_path in enumerate(images):
            # Get image info
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                width, height = 640, 480  # Default size
            
            # Add image info
            image_info = {
                "id": i + 1,
                "file_name": img_path.name,
                "width": width,
                "height": height
            }
            annotations["images"].append(image_info)
            
            # Add annotation (full image bounding box for now)
            annotation = {
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [0, 0, width, height],  # [x, y, width, height]
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []  # Empty for bounding box only
            }
            annotations["annotations"].append(annotation)
            annotation_id += 1
        
        # Save annotations
        ann_path = self.output_dir / f'{split}_annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(annotations, f, indent=2)
            
    def create_yolo_dataset_yaml(self):
        """Create YOLO dataset.yaml file"""
        yaml_content = f"""# YOLO dataset configuration for 5-shot images detection
path: {self.output_dir.absolute()}/yolo_format
real_dataset: real_dataset/images
val: val/images

nc: 1
names: ['images']

# 5-shot dataset configuration
n_shot: {self.n_shot}
seed: {self.seed}
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        return yaml_path


def create_sample_dataset():
    """Create a sample 5-shot dataset for testing"""
    print("🎨 Creating sample images images for testing...")
    
    # Create sample data directory
    sample_dir = Path("data/sample_polyps")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images images
    for i in range(20):  # Create 20 sample images
        img = Image.new('RGB', (640, 480), color='rgb(255, 255, 255)')
        draw = ImageDraw.Draw(img)
        
        # Draw random images-like shapes
        x1 = random.randint(50, 300)
        y1 = random.randint(50, 200)
        x2 = x1 + random.randint(50, 150)
        y2 = y1 + random.randint(50, 150)
        
        # Draw images (irregular shape)
        draw.ellipse([x1, y1, x2, y2], fill='rgb(200, 100, 100)', outline='rgb(150, 50, 50)')
        
        # Add some texture
        for _ in range(10):
            px = random.randint(x1, x2)
            py = random.randint(y1, y2)
            draw.point([px, py], fill='rgb(180, 80, 80)')
        
        img.save(sample_dir / f'polyp_{i:03d}.jpg')
    
    print(f"✅ Created {len(list(sample_dir.glob('*.jpg')))} sample images images")
    return sample_dir


if __name__ == "__main__":
    # Create sample dataset first
    sample_dir = create_sample_dataset()
    
    # Prepare 5-shot dataset
    preparer = FiveShotDatasetPreparer(
        source_dir=sample_dir,
        output_dir="data/5shot",
        n_shot=5
    )
    
    preparer.create_5shot_splits()
    preparer.create_yolo_dataset_yaml()
    
    print("🎯 5-shot dataset preparation complete!")
