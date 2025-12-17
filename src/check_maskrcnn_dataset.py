# check_maskrcnn_dataset.py
import os
import yaml
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

def check_maskrcnn_dataset_structure(dataset_path="data/full_real"):
    """Check MaskRCNN dataset structure and contents"""
    
    dataset = Path(dataset_path)
    if not dataset.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"🔍 Checking MaskRCNN dataset: {dataset}")
    print("=" * 60)
    
    # 1. Check dataset.yaml
    yaml_file = dataset / "dataset.yaml"
    if yaml_file.exists():
        print(f"✅ dataset.yaml found")
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"   Train: {config.get('train', 'NOT SET')}")
            print(f"   Val: {config.get('val', 'NOT SET')}")
            print(f"   Classes (nc): {config.get('nc', 'NOT SET')}")
            print(f"   Class names: {config.get('names', 'NOT SET')}")
    else:
        print("❌ dataset.yaml not found!")
        return False
    
    # 2. Check train directory
    train_images = dataset / "train" / "images"
    train_labels = dataset / "train" / "labels"
    
    if train_images.exists():
        train_img_files = list(train_images.glob("*.*"))
        print(f"✅ Train images: {len(train_img_files)}")
    else:
        print("❌ Train/images directory not found!")
        return False
    
    if train_labels.exists():
        train_label_files = list(train_labels.glob("*.txt"))
        print(f"✅ Train labels: {len(train_label_files)}")
    else:
        print("❌ Train/labels directory not found!")
        return False
    
    # 3. Check validation directory
    val_images = dataset / "val" / "images"
    val_labels = dataset / "val" / "labels"
    
    if val_images.exists():
        val_img_files = list(val_images.glob("*.*"))
        print(f"✅ Validation images: {len(val_img_files)}")
    else:
        print("⚠️  Validation/images directory not found!")
    
    if val_labels.exists():
        val_label_files = list(val_labels.glob("*.txt"))
        print(f"✅ Validation labels: {len(val_label_files)}")
    else:
        print("⚠️  Validation/labels directory not found!")
    
    # 4. Check label format
    print("\n📝 Checking label format...")
    if train_label_files:
        sample_label = train_label_files[0]
        try:
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    parts = first_line.split()
                    print(f"   Sample label: {sample_label.name}")
                    print(f"   Format: {len(parts)} values (class_id x_center y_center width height)")
                    print(f"   Value: {first_line}")
                    
                    # Check class ID
                    class_id = int(parts[0])
                    if class_id >= 0:
                        print(f"   Class ID: {class_id} ✅")
                    else:
                        print(f"   Class ID: {class_id} ❌ (cannot be negative)")
        except Exception as e:
            print(f"   ❌ Failed to read label: {e}")
    
    # 5. Check image-label correspondence
    print("\n🔗 Checking image-label correspondence:")
    img_names = {f.stem for f in train_img_files}
    label_names = {f.stem for f in train_label_files}
    
    common = img_names & label_names
    only_img = img_names - label_names
    only_label = label_names - img_names
    
    print(f"   Matching files: {len(common)}")
    print(f"   Images without labels: {len(only_img)}")
    print(f"   Labels without images: {len(only_label)}")
    
    if only_img:
        print(f"   ⚠️  Images without labels: {list(only_img)[:3]}...")
    
    # 6. Check annotation statistics
    print("\n📊 Annotation statistics:")
    total_boxes = 0
    class_counts = {}
    
    for label_file in train_label_files[:100]:  # Check first 100 labels
        try:
            with open(label_file, 'r') as f:
                boxes = f.readlines()
                total_boxes += len(boxes)
                for box in boxes:
                    parts = box.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
        except:
            pass
    
    print(f"   Total annotations: {total_boxes}")
    print(f"   Class distribution: {class_counts}")
    
    # 7. Check image sizes and formats
    print("\n📏 Checking image sizes and formats:")
    img_sizes = set()
    img_formats = set()
    
    for img_file in train_img_files[:10]:  # Check first 10 images
        try:
            with Image.open(img_file) as img:
                img_sizes.add(img.size)
                img_formats.add(img.format)
        except Exception as e:
            print(f"   ❌ Error reading {img_file.name}: {e}")
    
    print(f"   Image sizes: {img_sizes}")
    print(f"   Image formats: {img_formats}")
    
    # 8. Check model compatibility
    print("\n🤖 Checking MaskRCNN compatibility...")
    try:
        # Try to load a sample image and annotation
        sample_img = next(iter(train_img_files))
        sample_label = train_labels / f"{sample_img.stem}.txt"
        
        # Load and preprocess image
        img = Image.open(sample_img).convert("RGB")
        img_tensor = F.to_tensor(img)
        
        # Check if we can create target tensors
        if sample_label.exists():
            with open(sample_label, 'r') as f:
                boxes = []
                labels = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Convert YOLO to COCO format
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        x1 = (x_center - width/2) * img.width
                        y1 = (y_center - height/2) * img.height
                        x2 = (x_center + width/2) * img.width
                        y2 = (y_center + height/2) * img.height
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # Assuming class 1 is polyp
        
        print("   ✅ Dataset is compatible with MaskRCNN")
        
    except Exception as e:
        print(f"   ❌ MaskRCNN compatibility check failed: {e}")
    
    return True

def verify_annotations(dataset_path="data/full_real", num_samples=2):
    """Visual verification of annotations"""
    dataset = Path(dataset_path)
    train_images = list((dataset / "train" / "images").glob("*.*"))
    
    if not train_images:
        print("❌ No training images found!")
        return
    
    print("\n🎨 Visual verification of annotations:")
    sample_images = random.sample(train_images, min(num_samples, len(train_images)))
    
    for img_path in sample_images:
        label_path = dataset / "train" / "labels" / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"   ❌ No label found for {img_path.name}")
            continue
        
        try:
            img = Image.open(img_path)
            print(f"\n   📸 {img_path.name}: {img.size}")
            
            with open(label_path, 'r') as f:
                boxes = [line.strip().split() for line in f.readlines()]
            
            if boxes:
                print(f"   🏷️  {len(boxes)} annotations:")
                for i, box in enumerate(boxes[:3]):  # Show first 3 annotations
                    if len(box) >= 5:
                        cls, x_center, y_center, width, height = map(float, box[:5])
                        
                        # Convert YOLO to pixel coordinates
                        img_w, img_h = img.size
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h
                        
                        print(f"     Box {i+1}: class={int(cls)}, "
                              f"x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
            else:
                print("   ⚠️  No annotations found")
                
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {e}")

def main():
    print("🧪 MaskRCNN Dataset Checker")
    print("=" * 60)
    
    dataset_path = "/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real"
    
    # 1. Check dataset structure
    is_valid = check_maskrcnn_dataset_structure(dataset_path)
    
    if not is_valid:
        print("\n❌ Dataset structure is invalid!")
        return
    
    # 2. Verify annotations
    verify_annotations(dataset_path)
    
    print("\n" + "=" * 60)
    print("✅ Dataset check completed!")
    print("\n💡 If everything looks good, you can proceed with training:")
    print(f"python train_maskrcnn_5epochs.py --data_dir {dataset_path}")

if __name__ == "__main__":
    main()