# check_yolo_dataset.py
import os
from pathlib import Path
import yaml
from PIL import Image
import random

def check_yolo_dataset_structure(dataset_path="/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real"):
    """YOLO dataset strukturasi va mazmunini tekshirish"""
    
    dataset = Path(dataset_path)
    if not dataset.exists():
        print(f"❌ Dataset topilmadi: {dataset_path}")
        return False
    
    print(f"🔍 Datasetni tekshirish: {dataset}")
    print("=" * 60)
    
    # 1. dataset.yaml faylini tekshirish
    yaml_file = dataset / "dataset.yaml"
    if yaml_file.exists():
        print(f"✅ dataset.yaml mavjud")
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"   Train: {config.get('train', 'NOT SET')}")
            print(f"   Val: {config.get('val', 'NOT SET')}")
            print(f"   Classes (nc): {config.get('nc', 'NOT SET')}")
            print(f"   Class names: {config.get('names', 'NOT SET')}")
    else:
        print("❌ dataset.yaml yo'q!")
        return False
    
    # 2. Train folderini tekshirish
    train_images = dataset / "train" / "images"
    train_labels = dataset / "train" / "labels"
    
    if train_images.exists():
        train_img_files = list(train_images.glob("*.*"))
        print(f"✅ Train images: {len(train_img_files)} ta")
    else:
        print("❌ Train/images folderi yo'q!")
        return False
    
    if train_labels.exists():
        train_label_files = list(train_labels.glob("*.txt"))
        print(f"✅ Train labels: {len(train_label_files)} ta")
    else:
        print("❌ Train/labels folderi yo'q!")
        return False
    
    # 3. Val folderini tekshirish
    val_images = dataset / "val" / "images"
    val_labels = dataset / "val" / "labels"
    
    if val_images.exists():
        val_img_files = list(val_images.glob("*.*"))
        print(f"✅ Val images: {len(val_img_files)} ta")
    else:
        print("⚠️ Val/images folderi yo'q!")
    
    if val_labels.exists():
        val_label_files = list(val_labels.glob("*.txt"))
        print(f"✅ Val labels: {len(val_label_files)} ta")
    else:
        print("⚠️ Val/labels folderi yo'q!")
    
    # 4. Label formatini tekshirish
    print("\n📝 Label formatini tekshirish...")
    if train_label_files:
        sample_label = train_label_files[0]
        try:
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    parts = first_line.split()
                    print(f"   Namuna label: {sample_label.name}")
                    print(f"   Format: {len(parts)} ta raqam (class_id x_center y_center width height)")
                    print(f"   Qiymat: {first_line}")
                    
                    # Class id ni tekshirish
                    class_id = int(parts[0])
                    if class_id >= 0:
                        print(f"   Class ID: {class_id} ✅")
                    else:
                        print(f"   Class ID: {class_id} ❌ (manfiy bo'lishi mumkin emas)")
        except Exception as e:
            print(f"   ❌ Label o'qib bo'lmadi: {e}")
    
    # 5. Rasmlar va label'lar mosligini tekshirish
    print("\n🔗 Rasmlar va label'lar mosligi:")
    img_names = {f.stem for f in train_img_files}
    label_names = {f.stem for f in train_label_files}
    
    common = img_names & label_names
    only_img = img_names - label_names
    only_label = label_names - img_names
    
    print(f"   Mos keladigan fayllar: {len(common)} ta")
    print(f"   Faqat rasm bor: {len(only_img)} ta")
    print(f"   Faqat label bor: {len(only_label)} ta")
    
    if only_img:
        print(f"   ⚠️ Label'siz rasmlar: {list(only_img)[:3]}...")
    
    # 6. Annotation statistikasi
    print("\n📊 Annotation statistikasi:")
    total_boxes = 0
    class_counts = {}
    
    for label_file in train_label_files[:10]:  # Birinchi 10 ta label
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
    
    print(f"   Train annotation'lar: {total_boxes} ta")
    print(f"   Class bo'yicha taqsimot: {class_counts}")
    
    # 7. Rasm o'lchamlarini tekshirish
    print("\n📏 Rasm o'lchamlari:")
    sizes = set()
    for img_file in train_img_files[:5]:
        try:
            with Image.open(img_file) as img:
                sizes.add(img.size)
        except:
            pass
    
    print(f"   Turli o'lchamlar: {sizes}")
    
    return True

def fix_dataset_yaml(dataset_path="data/full_real"):
    """dataset.yaml faylini to'g'rilash"""
    
    dataset = Path(dataset_path)
    yaml_file = dataset / "dataset.yaml"
    
    # Relative yo'llarni absolute ga o'girish
    train_path = str((dataset / "train" / "images").absolute())
    val_path = str((dataset / "val" / "images").absolute())
    
    # YAML content
    yaml_content = {
        'path': str(dataset.absolute()),
        'train': train_path,
        'val': val_path,
        'nc': 1,  # Faqat polyp class
        'names': ['polyp']
    }
    
    # Yozish
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✅ dataset.yaml yangilandi:")
    print(f"   path: {yaml_content['path']}")
    print(f"   train: {yaml_content['train']}")
    print(f"   val: {yaml_content['val']}")
    print(f"   nc: {yaml_content['nc']}")
    print(f"   names: {yaml_content['names']}")
    
    return yaml_file

def verify_annotations(dataset_path="data/full_real"):
    """Annotation'larni vizual tekshirish"""
    
    dataset = Path(dataset_path)
    train_images = list((dataset / "train" / "images").glob("*.*"))
    
    if not train_images:
        print("❌ Train rasmlari yo'q!")
        return
    
    # 2 ta random rasm tanlash
    sample_images = random.sample(train_images, min(2, len(train_images)))
    
    print("\n🎨 Annotation'larni vizual tekshirish:")
    
    for img_path in sample_images:
        label_path = dataset / "train" / "labels" / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"   ❌ {img_path.name} uchun label yo'q")
            continue
        
        # Rasmni ochish
        try:
            img = Image.open(img_path)
            print(f"\n   📸 {img_path.name}: {img.size}")
            
            # Annotation'larni o'qish
            with open(label_path, 'r') as f:
                boxes = [line.strip().split() for line in f.readlines()]
            
            if boxes:
                print(f"   🏷️  {len(boxes)} ta annotation:")
                for i, box in enumerate(boxes[:3]):  # Birinchi 3 tasi
                    cls, x_center, y_center, width, height = map(float, box)
                    
                    # YOLO formatdan pixel formatga o'tkazish
                    img_w, img_h = img.size
                    x_center_px = x_center * img_w
                    y_center_px = y_center * img_h
                    width_px = width * img_w
                    height_px = height * img_h
                    
                    # Bounding box koordinatalari
                    x1 = x_center_px - width_px / 2
                    y1 = y_center_px - height_px / 2
                    x2 = x_center_px + width_px / 2
                    y2 = y_center_px + height_px / 2
                    
                    print(f"     Box {i+1}: class={int(cls)}, "
                          f"x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
            else:
                print("   ⚠️  Hech qanday annotation yo'q")
                
        except Exception as e:
            print(f"   ❌ {img_path.name}: {e}")

def main():
    print("🧪 YOLO Datasetni tekshirish va tuzatish")
    print("=" * 60)
    
    # 1. Datasetni tekshirish
    is_valid = check_yolo_dataset_structure("/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real")
    
    if not is_valid:
        print("\n❌ Dataset strukturasi noto'g'ri!")
        return
    
    print("\n" + "=" * 60)
    print("🔧 Datasetni tuzatish...")
    
    # 2. dataset.yaml ni tuzatish
    fix_dataset_yaml("/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real")
    
    # 3. Annotation'larni tekshirish
    verify_annotations("/Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real")
    
    # 4. Train uchun tayyor
    print("\n" + "=" * 60)
    print("✅ Dataset train qilish uchun tayyor!")
    print("\nTrain qilish uchun:")
    print("python train_yolo.py --data /Users/azimjonakromov/Downloads/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/data/full_real/dataset.yaml")

if __name__ == "__main__":
    main()