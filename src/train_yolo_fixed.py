# train_yolo_fixed.py
import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import argparse
import shutil

def prepare_dataset():
    """Datasetni train uchun tayyorlash"""
    
    # Dataset yo'llari
    dataset_path = Path("/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real")
    yaml_file = dataset_path / "dataset.yaml"
    
    # dataset.yaml ni tekshirish va to'g'rilash
    if not yaml_file.exists():
        print("❌ dataset.yaml yo'q, yaratilmoqda...")
        
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': str((dataset_path / "train" / "images").absolute()),
            'val': str((dataset_path / "val" / "images").absolute()),
            'nc': 1,
            'names': ['polyp']
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        print(f"✅ dataset.yaml yaratildi: {yaml_file}")
    
    # Datasetni tekshirish
    train_images = list((dataset_path / "train" / "images").glob("*.*"))
    train_labels = list((dataset_path / "train" / "labels").glob("*.txt"))
    
    val_images = list((dataset_path / "val" / "images").glob("*.*"))
    val_labels = list((dataset_path / "val" / "labels").glob("*.txt"))
    
    print(f"📊 Dataset statistikasi:")
    print(f"   Train rasmlari: {len(train_images)}")
    print(f"   Train label'lar: {len(train_labels)}")
    print(f"   Val rasmlari: {len(val_images)}")
    print(f"   Val label'lar: {len(val_labels)}")
    
    # Label formatini tekshirish
    if train_labels:
        with open(train_labels[0], 'r') as f:
            sample = f.readline().strip()
            print(f"   Label format namunasi: {sample}")
    
    return str(yaml_file)

def train_model(data_yaml, model_size='n', epochs=5, imgsz=640):
    """YOLOv8 modelini train qilish"""
    
    # Model nomi
    model_name = f'yolov8{model_size}.pt'
    print(f"\n🚀 Train boshlanmoqda...")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Modelni yuklash
        model = YOLO(model_name)
        
        # Train parametrlari
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=30,  # Early stopping uchun
            save=True,
            save_period=5,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=4,
            project='polyp_detection',
            name=f'full_real_yolov8{model_size}',
            exist_ok=True,
            verbose=True,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            # Learning rate
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0
        )
        
        print(f"\n✅ Train muvaffaqiyatli yakunlandi!")
        
        # Best model yo'li
        best_model = Path(f"polyp_detection/full_real_yolov8{model_size}/weights/best.pt")
        if best_model.exists():
            print(f"📁 Eng yaxshi model: {best_model}")
            
            # Modelni validate qilish
            print("\n🔍 Modelni validate qilish...")
            metrics = model.val(
                data=data_yaml,
                split='val',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            
            print(f"📊 Validation natijalari:")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   Precision: {metrics.box.p:.4f}")
            print(f"   Recall: {metrics.box.r:.4f}")
            
            return str(best_model)
        else:
            print("⚠️ Best model fayli topilmadi")
            return None
            
    except Exception as e:
        print(f"❌ Train jarayonida xato: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trained_model(model_path, test_images_dir="data/full_real/val/images"):
    """Train qilingan modelni test qilish"""
    
    if not Path(model_path).exists():
        print(f"❌ Model topilmadi: {model_path}")
        return
    
    print(f"\n🧪 Modelni test qilish...")
    
    # Modelni yuklash
    model = YOLO(model_path)
    
    # Test rasmlari
    test_images = list(Path(test_images_dir).glob("*.*"))
    if not test_images:
        print(f"⚠️ Test rasmlari topilmadi: {test_images_dir}")
        return
    
    # Bir nechta test rasmlari
    for i, img_path in enumerate(test_images[:3]):
        print(f"\nTest {i+1}: {img_path.name}")
        
        # Prediction
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            save=True,
            save_txt=True,
            save_conf=True,
            project='polyp_test',
            name='full_real',
            exist_ok=True
        )
        
        # Natijalarni chiqarish
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                print(f"   Detected: {len(boxes)} ta polyp")
                for j, box in enumerate(boxes):
                    conf = box.conf.item()
                    cls = box.cls.item()
                    xyxy = box.xyxy[0].tolist()
                    print(f"     Box {j+1}: class={cls}, conf={conf:.3f}, "
                          f"coords={[int(x) for x in xyxy]}")
            else:
                print("   Hech qanday polyp aniqlanmadi")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Polyp Detection Train')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (nano, small, medium, large, xlarge)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing model')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Faqat test qilish
        model_path = input("Test qilish uchun model yo'lini kiriting: ").strip()
        if model_path:
            test_trained_model(model_path)
        return
    
    # 1. Datasetni tayyorlash
    print("📁 Datasetni tayyorlash...")
    data_yaml = prepare_dataset()
    
    # 2. Model train qilish
    trained_model = train_model(
        data_yaml=data_yaml,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz
    )
    
    # 3. Test qilish
    if trained_model:
        test_trained_model(trained_model)
        
        print(f"\n🎉 Barcha jarayonlar muvaffaqiyatli yakunlandi!")
        print(f"📁 Model: {trained_model}")
        print(f"📊 Natijalar: polyp_detection/full_real_yolov8{args.model}/")
        print(f"\nGUI uchun fayl: src/gui_fixed.py")

if __name__ == "__main__":
    main()