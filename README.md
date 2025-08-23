# DEXI YOLO Training Pipeline

A complete training pipeline for custom YOLO models with rotation and scale invariance, designed specifically for drone-based object detection.

## Features

- **Rotation-invariant training**: 360° rotation augmentation for drone perspectives
- **Scale-invariant training**: Multi-scale augmentation for varying altitudes
- **Complete data pipeline**: From raw images to trained models
- **Optimized for deployment**: Supports conversion to ONNX for edge devices
- **Comprehensive augmentation**: Lighting, noise, blur, and geometric transforms

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python numpy
```

### 2. Prepare Your Images

Place your base images in a folder (e.g., `base_images/`). Name them clearly:
- `bird.jpg`, `dog.jpg`, `cat.jpg`, `motorcycle.jpg`, `car.jpg`, `truck.jpg`

### 3. Generate Training Dataset

```bash
python augment_dataset.py --input base_images/ --count 150
```

This generates 150 augmented variants per image with:
- 360° rotations
- Scale variations (0.7x - 1.3x)  
- Brightness/contrast changes
- Gaussian noise and blur
- Proper YOLO bounding box labels

### 4. Train Model

```bash
python train_yolo.py --model n --epochs 50 --batch 8
```

Models available: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge)

## File Structure

```
dexi_yolo_train/
├── augment_dataset.py     # Data augmentation script
├── train_yolo.py         # Training script  
├── dataset.yaml          # YOLO dataset configuration
├── train/               # Training images and labels
├── val/                 # Validation images and labels
├── runs/               # Training outputs and metrics
└── README.md
```

## Training Configuration

- **Classes**: 6 custom classes (bird, dog, cat, motorcycle, car, truck)
- **Base Model**: YOLOv8 with COCO pre-trained weights
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Augmentation**: Rotation-invariant with minimal additional transforms during training
- **Output**: PyTorch (.pt) models ready for deployment

## Model Deployment

After training, your best model will be saved to:
```
runs/detect/drone_detection*/weights/best.pt
```

### Convert to ONNX (for deployment)
```python
from ultralytics import YOLO
model = YOLO('runs/detect/drone_detection/weights/best.pt')
model.export(format='onnx')
```

## Integration with DEXI

This training pipeline is designed to work with the DEXI YOLO ROS2 node. Replace the default model with your trained model for custom object detection.

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

## License

Licensed under the same terms as the DEXI project.

## Contributing

This pipeline is part of the DEXI ecosystem. For issues and contributions, please refer to the main DEXI project guidelines.