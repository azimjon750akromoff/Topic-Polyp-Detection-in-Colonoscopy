import argparse
import torch
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.yolo_baseline import YOLOv8Baseline
from src.models.maskrcnn_baseline import MaskRCNNBaseline, PolypDataset
from src.prepare_real_dataset import prepare_real_5shot_dataset


def train_yolo_baseline(args):
    """Train YOLOv8 baseline model"""
    print("🚀 Training YOLOv8 baseline...")
    
    # Initialize YOLO model
    yolo = YOLOv8Baseline(model_size=args.model_size, device=args.device)
    yolo.load_model()
    
    # Prepare dataset if needed
    if args.prepare_dataset:
        print("📊 Preparing 5-shot dataset from real polyp images...")
        dataset_dir = prepare_real_5shot_dataset()
        data_yaml = dataset_dir / 'dataset.yaml'
    else:
        data_yaml = Path(args.data_dir) / 'dataset.yaml'
    
    # Train model
    results = yolo.train_5shot(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size
    )
    
    # Evaluate model
    metrics = yolo.evaluate(data_yaml=data_yaml, imgsz=args.img_size)
    
    # Save results
    results_dict = {
        'model': 'YOLOv8',
        'model_size': args.model_size,
        'n_shot': args.n_shot,
        'epochs': args.epochs,
        'metrics': metrics,
        'training_time': time.time()
    }
    
    with open('experiments/results/yolo_baseline_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("✅ YOLOv8 baseline training completed!")
    return metrics


def train_maskrcnn_baseline(args):
    """Train Mask R-CNN baseline model"""
    print("🚀 Training Mask R-CNN baseline...")
    
    # Initialize Mask R-CNN model
    maskrcnn = MaskRCNNBaseline(num_classes=2, device=args.device)
    maskrcnn.load_model()
    
    # Prepare dataset if needed
    if args.prepare_dataset:
        print("📊 Preparing 5-shot dataset from real polyp images...")
        dataset_dir = prepare_real_5shot_dataset()
        data_dir = dataset_dir
    else:
        data_dir = Path(args.data_dir)
    
    # Create datasets
    train_dataset = PolypDataset(
        root=data_dir / 'train' / 'images',
        transforms=maskrcnn.get_transform(train=True)
    )
    
    val_dataset = PolypDataset(
        root=data_dir / 'val' / 'images',
        transforms=maskrcnn.get_transform(train=False)
    )
    
    print(f"📊 Dataset size:")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    
    # Train model
    best_map = maskrcnn.train_5shot(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Create dummy data loader for evaluation
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=maskrcnn._collate_fn
    )
    
    # Evaluate model
    metrics = maskrcnn.evaluate(data_loader=val_loader)
    
    # Save results
    results_dict = {
        'model': 'MaskRCNN',
        'n_shot': args.n_shot,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'metrics': metrics,
        'best_map': best_map,
        'training_time': time.time()
    }
    
    with open('experiments/results/maskrcnn_baseline_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("✅ Mask R-CNN baseline training completed!")
    return metrics


def compare_baselines():
    """Compare baseline results"""
    print("📊 Comparing baseline models...")
    
    results = {}
    
    # Load YOLO results
    try:
        with open('experiments/results/yolo_baseline_results.json', 'r') as f:
            results['YOLOv8'] = json.load(f)
    except FileNotFoundError:
        print("⚠️  YOLOv8 results not found")
    
    # Load Mask R-CNN results
    try:
        with open('experiments/results/maskrcnn_baseline_results.json', 'r') as f:
            results['MaskRCNN'] = json.load(f)
    except FileNotFoundError:
        print("⚠️  Mask R-CNN results not found")
    
    if not results:
        print("❌ No baseline results found!")
        return
    
    # Create comparison table
    print("\n📈 Baseline Comparison:")
    print("-" * 60)
    print(f"{'Model':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<12} {metrics['map50']:<10.4f} {metrics['map50_95']:<12.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    # Plot comparison
    if len(results) > 1:
        plot_comparison(results)
    
    return results


def plot_comparison(results):
    """Plot baseline comparison"""
    models = list(results.keys())
    metrics = ['map50', 'map50_95', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Baseline Models Comparison (5-shot)', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [results[model]['metrics'][metric] for model in models]
        
        bars = ax.bar(models, values, alpha=0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiments/results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plot saved to experiments/results/baseline_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Train baseline models for 5-shot polyp detection')
    parser.add_argument('--model', choices=['yolo', 'maskrcnn', 'both'], default='both',
                       help='Which model to train')
    parser.add_argument('--data_dir', type=str, default='data/5shot',
                       help='Dataset directory')
    parser.add_argument('--n_shot', type=int, default=5,
                       help='Number of examples per class')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for YOLO')
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--prepare_dataset', action='store_true',
                       help='Prepare 5-shot dataset from scratch')
    parser.add_argument('--compare', action='store_true',
                       help='Compare baseline results')
    
    args = parser.parse_args()
    
    # Create experiments directories
    Path('experiments/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('experiments/results').mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        compare_baselines()
        return
    
    results = {}
    
    if args.model in ['yolo', 'both']:
        try:
            results['YOLOv8'] = train_yolo_baseline(args)
        except Exception as e:
            print(f"❌ YOLOv8 training failed: {e}")
    
    if args.model in ['maskrcnn', 'both']:
        try:
            results['MaskRCNN'] = train_maskrcnn_baseline(args)
        except Exception as e:
            print(f"❌ Mask R-CNN training failed: {e}")
    
    if len(results) > 1:
        # Auto-compare if both models were trained
        compare_baselines()
    
    print("\n🎉 Baseline training completed!")
    print(f"Results saved to experiments/results/")


if __name__ == "__main__":
    main()
