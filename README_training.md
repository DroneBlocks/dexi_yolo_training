# YOLO Training Setup for Drone Detection

## Quick Start

1. **Add your 6 base images** to a folder (e.g., `base_images/`)
   - Name them clearly: `bird.jpg`, `dog.jpg`, `cat.jpg`, `motorcycle.jpg`, `car.jpg`, `truck.jpg`

2. **Generate augmented dataset:**
   ```bash
   python augment_dataset.py --input base_images/ --count 100
   ```

3. **Install ultralytics:**
   ```bash
   pip install ultralytics
   ```

4. **Train the model:**
   ```bash
   python train_yolo.py --model n --epochs 100
   ```

## Data Augmentation Features

The augmentation script generates variations with:
- **360° rotations** (critical for drone orientation independence)
- **Scale variations** (0.7x to 1.3x)
- **Brightness/contrast changes**
- **Gaussian noise and blur**
- **Proper YOLO bounding box labels**

## Training Configuration

- **6 classes:** bird, dog, cat, motorcycle, car, truck
- **Pre-trained YOLOv8 base:** Starts with COCO weights
- **Optimized for single objects:** Each image contains one centered object
- **Rotation-invariant:** No additional rotation augmentation during training

## Model Sizes

- `n` (nano): Fastest, smallest (~3MB)
- `s` (small): Balanced (~11MB)
- `m` (medium): More accurate (~25MB)

For drone deployment, start with nano model for speed.

## Expected Results

With 100 augmentations per class (600 total images), you should achieve:
- High accuracy on your specific 6 classes
- Rotation invariance up to 360°
- Scale invariance for drone altitude changes
- Good performance in various lighting conditions