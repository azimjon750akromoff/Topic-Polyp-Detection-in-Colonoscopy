#!/usr/bin/env python3
"""
Train Mask R-CNN for 25 epochs on full_real dataset
Similar to YOLOv8 training that was done yesterday
"""
import sys
from pathlib import Path
import torch
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.maskrcnn_baseline import MaskRCNNBaseline, PolypDataset


def train_maskrcnn_25epochs(data_dir="data/full_real", epochs=25, batch_size=4, lr=0.001):
    """
    Train Mask R-CNN for 25 epochs on full_real dataset
    
    Args:
        data_dir: Path to dataset directory (should have train/ and val/ subdirectories)
        epochs: Number of training epochs (default: 25)
        batch_size: Batch size for training
        lr: Learning rate
    """
    print("=" * 60)
    print("🚀 MASK R-CNN TRAINING - 25 EPOCHS")
    print("=" * 60)
    
    # Check dataset path
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Dataset path not found: {data_path}")
        print(f"💡 Please provide correct path to full_real dataset")
        return None
    
    # Check for train and val directories
    train_images_dir = data_path / "train" / "images"
    val_images_dir = data_path / "val" / "images"
    
    if not train_images_dir.exists():
        print(f"❌ Train images directory not found: {train_images_dir}")
        return None
    
    if not val_images_dir.exists():
        print(f"❌ Val images directory not found: {val_images_dir}")
        return None
    
    print(f"📁 Dataset: {data_path}")
    print(f"   Train: {train_images_dir}")
    print(f"   Val: {val_images_dir}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Device: {device.upper()}")
    
    # Initialize Mask R-CNN model
    print("\n🤖 Loading Mask R-CNN model...")
    # num_classes=1 means 1 class (polyp) + 1 background = 2 total classes for the model
    maskrcnn = MaskRCNNBaseline(num_classes=1, device=device)  # 1 class: polyp
    maskrcnn.load_model()
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = PolypDataset(
        root=train_images_dir,
        transforms=maskrcnn.get_transform(train=True)
    )
    
    val_dataset = PolypDataset(
        root=val_images_dir,
        transforms=maskrcnn.get_transform(train=False)
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    
    if len(train_dataset) == 0:
        print("❌ No training images found!")
        return None
    
    # Create checkpoint directory
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Training parameters:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    # Train model
    print("\n🚀 Starting training...")
    try:
        best_map = maskrcnn.train_5shot(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )
        
        print(f"\n✅ Training completed!")
        print(f"   Best mAP: {best_map:.4f}")
        
        # Save final model
        final_model_path = checkpoint_dir / "maskrcnn_25epochs_best.pt"
        maskrcnn.save_model(str(final_model_path))
        print(f"   Model saved: {final_model_path}")
        
        # Evaluate
        print("\n📊 Evaluating model...")
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=maskrcnn._collate_fn
        )
        
        metrics = maskrcnn.evaluate(data_loader=val_loader)
        
        print(f"\n📈 Final Metrics:")
        print(f"   mAP@0.5: {metrics.get('map50', 0):.4f}")
        print(f"   mAP@0.5:0.95: {metrics.get('map50_95', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        
        print(f"\n🎉 Mask R-CNN training completed successfully!")
        print(f"📁 Model saved to: {final_model_path}")
        print(f"\n💡 To use this model in GUI, update the model loading path in gui.py")
        
        return final_model_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for 25 epochs')
    parser.add_argument('--data_dir', type=str, default='data/full_real',
                       help='Path to dataset directory (default: data/full_real)')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    train_maskrcnn_25epochs(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()