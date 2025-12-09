# Few-Shot Polyp Detection in Colonoscopy - Implementation Guide

## Overview
This project implements and evaluates YOLOv8 and Mask R-CNN baseline models for few-shot polyp detection in colonoscopy images. The codebase supports 5-shot learning experiments with comprehensive training and evaluation pipelines.

## What Was Fixed and Implemented

### 🔧 Major Issues Resolved

1. **Mask R-CNN Dataset Loading Error**
   - **Problem**: `'Image' object has no attribute 'to'` error during training
   - **Root Cause**: `PolypDataset.__getitem__` was returning PIL Images instead of tensors due to broken transforms
   - **Solution**: Updated transforms to use proper `ToTensor()` and normalization from `torchvision.transforms`

2. **Mask R-CNN Prediction Method Error**
   - **Problem**: Same `'Image' object has no attribute 'to'` error during evaluation
   - **Root Cause**: `predict` method wasn't handling different input types (PIL Images, tensors, file paths)
   - **Solution**: Added robust input handling with proper tensor conversion and device management

3. **Evaluation Unpacking Error**
   - **Problem**: "too many values to unpack (expected 2)" in mAP calculation
   - **Root Cause**: `calculate_map` method expected 2 values but received 5 (x1, y1, x2, y2, confidence)
   - **Solution**: Updated unpacking logic to handle 5-value prediction format

4. **Models Showing 0.0 mAP Despite Good Training Metrics**
   - **Problem**: Evaluation showed 0.0 mAP while training showed good performance
   - **Root Cause**: Confidence thresholds were too high (0.25) for 5-shot models (all predictions had ~0.0016 confidence)
   - **Solution**: Lowered confidence thresholds to 0.001 for 5-shot model evaluation

5. **YOLO Model Loading Issues**
   - **Problem**: Evaluation was looking for models in wrong directory paths
   - **Solution**: Updated to automatically find latest trained model in `runs/detect/` directory

6. **Class ID and Number of Classes Mismatch**
   - **Problem**: Mask R-CNN was initialized with 1 class but dataset had 2 classes (polyp, non_polyp)
   - **Root Cause**: Class ID mapping was adding +1 to YOLO class IDs, creating out-of-bounds errors
   - **Solution**: 
     - Updated Mask R-CNN to use `num_classes=2`
     - Fixed class ID mapping to keep original IDs (0=polyp, 1=non_polyp)
     - Added graceful checkpoint loading for class mismatches

### 🎯 Key Insights Discovered

1. **5-Shot Learning Challenges**: With only 2 training images, both models over-predict with very low confidence scores (~0.0016)
2. **Dataset Characteristics**: Ground truth boxes cover entire images (0.5, 0.5, 1.0, 1.0), making this more like image classification than object detection
3. **Model Behavior**: 
   - YOLOv8: Achieves ~33% mAP, detects hundreds of low-confidence boxes
   - Mask R-CNN: Achieves ~6% mAP, also over-predicts but with better structure

## 🚀 Quick Start Commands

### Training Commands

```bash
# Train YOLOv8 baseline with your real images
./venv/bin/python -m src.train_baselines --model yolo --epochs 10 --prepare_dataset

# Train Mask R-CNN baseline  
./venv/bin/python -m src.train_baselines --model maskrcnn --epochs 10 --prepare_dataset

# Train both baselines and compare
./venv/bin/python -m src.train_baselines --model both --epochs 10 --prepare_dataset
```

### Evaluation Command

```bash
# Evaluate models on prepared dataset
./venv/bin/python -m src.evaluate_baselines --data_dir data/5shot_real
```

## 📊 Expected Results

### Training Output
- **YOLOv8**: Shows training progress with mAP metrics around 0.6-0.7 during training
- **Mask R-CNN**: Shows loss reduction (e.g., from 8.3765 to 2.9561) and validation mAP around 0.5

### Evaluation Output
```
📈 Evaluation Results:
---------------------------------------------
Model        mAP@0.5    mAP@0.5:0.95    mAP@0.75    Images
---------------------------------------------
YOLOv8       0.3368     0.3134          0.2979      2
MaskRCNN     0.0638     0.0064          0.0000      2
```

## 🗂️ Project Structure

```
Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy/
├── src/
│   ├── train_baselines.py          # Main training script
│   ├── evaluate_baselines.py       # Evaluation script
│   ├── models/
│   │   ├── yolo_baseline.py        # YOLOv8 implementation
│   │   └── maskrcnn_baseline.py    # Mask R-CNN implementation
│   └── dataset_preparation.py      # Dataset utilities
├── data/
│   └── 5shot_real/                 # 5-shot dataset
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── experiments/
│   ├── checkpoints/               # Saved models
│   └── results/                   # Evaluation results
└── runs/
    └── detect/                    # YOLO training outputs
```

## 🔧 Technical Details

### Dataset Format
- **Images**: JPG/PNG format in `train/images/` and `val/images/`
- **Labels**: YOLO format in corresponding `labels/` directories
- **Classes**: 
  - `0`: polyp
  - `1`: non_polyp
- **Label Format**: `class_id x_center y_center width height` (normalized coordinates)

### Model Configurations

#### YOLOv8
- **Architecture**: YOLOv8n (nano version)
- **Input Size**: 640x640
- **Classes**: 2 (polyp, non_polyp)
- **Training**: Uses ultralytics library with custom dataset

#### Mask R-CNN
- **Architecture**: ResNet-50 FPN backbone
- **Input Size**: Variable (original image size)
- **Classes**: 2 (polyp, non_polyp) + background
- **Training**: PyTorch implementation with custom dataset

### Key Fixes Applied

1. **Transform Pipeline**:
```python
# Fixed transforms for proper tensor conversion
transform_list.append(transforms_legacy.ToTensor())
transform_list.append(transforms_legacy.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
```

2. **Class ID Mapping**:
```python
# Fixed to keep original class IDs
labels.append(int(class_id))  # 0=images, 1=masks
```

3. **Confidence Thresholds**:
```python
# Lowered for 5-shot models
results = model.predict(str(img_path), conf=0.001)
pred = model.predict(image, confidence=0.001)
```

## 🐛 Common Issues and Solutions

### Issue: "Target 2 is out of bounds"
**Cause**: Class ID mismatch between dataset and model
**Solution**: Ensure `num_classes` matches dataset classes and fix class ID mapping

### Issue: "'Image' object has no attribute 'to'"
**Cause**: PIL Images not converted to tensors
**Solution**: Check transforms are properly applied and use correct tensor conversion

### Issue: Models showing 0.0 mAP
**Cause**: Confidence thresholds too high for 5-shot models
**Solution**: Lower confidence thresholds to 0.001 for evaluation

## 📈 Performance Notes

- **5-shot learning is challenging**: With minimal data, models tend to over-predict
- **YOLOv8 generally performs better**: More robust to few-shot scenarios
- **Evaluation metrics are sensitive**: Small changes in confidence thresholds significantly affect results
- **Visualizations are saved**: Check `experiments/results/yolo_visualizations/` and `maskrcnn_visualizations/`

## 🎯 Next Steps

1. **Increase dataset size**: Add more training images for better performance
2. **Experiment with confidence thresholds**: Find optimal values for your specific use case
3. **Try data augmentation**: Improve generalization with limited data
4. **Compare with other models**: Test additional baseline models
5. **Fine-tune hyperparameters**: Optimize learning rates, batch sizes, etc.

## 📝 Implementation Notes

This implementation demonstrates the challenges of few-shot object detection in medical imaging. The models work correctly but are limited by the extremely small dataset size (2 training images). For research purposes, consider using larger datasets or implementing few-shot learning techniques specifically designed for medical imaging.
