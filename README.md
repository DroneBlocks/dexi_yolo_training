# DEXI YOLO Training Pipeline for Drone Detection

## 🚀 Quick Start

### **Method 1: Google Colab (Easiest - No Setup Required)**

🌟 **Run in Google Colab with free GPU access:**

1. **Open the Colab notebook:**
   - Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/droneblocks/dexi_yolo_training/blob/main/YOLO_Training_Tutorial_Colab.ipynb)
   - Or go to [Google Colab](https://colab.research.google.com/) and upload `YOLO_Training_Tutorial_Colab.ipynb`

2. **Enable GPU for faster training:**
   - Go to `Runtime > Change runtime type > Hardware accelerator > GPU`

3. **Upload your 6 base images:**
   - Use the file upload widget in the notebook
   - Or drag-drop files into Colab's file browser

4. **Run all cells** - the notebook handles everything automatically!

**🚀 Colab Benefits:**
- ✅ **Free Tesla T4 GPU** (10-20x faster than CPU)
- ✅ **No local setup required** (just a browser)
- ✅ **12GB RAM + 100GB storage** (plenty for training)
- ✅ **Pre-installed ML libraries** (PyTorch, OpenCV, etc.)
- ✅ **Easy sharing** (send link to collaborators)
- ✅ **Automatic model download** (best.pt and ONNX files)

## 📊 **Which Notebook Should I Use?**

| Feature | Google Colab | Local Jupyter |
|---------|-------------|---------------|
| **Setup** | None required | Virtual env + packages |
| **GPU** | Free Tesla T4 | Your own GPU/CPU |
| **Training Speed** | 15-25 min (GPU) | Varies by hardware |
| **File Management** | Upload/download | Direct file access |
| **Internet** | Required | Optional |
| **Best For** | Beginners, no GPU | Advanced users, privacy |

**🎯 Recommendation**: Start with **Colab** for easiest experience!

### **Method 2: Local Jupyter Notebook**

1. **Setup environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Launch the local tutorial:**
   ```bash
   jupyter notebook YOLO_Training_Tutorial_Local.ipynb
   ```

3. **Follow the interactive guide** that walks you through:
   - Dataset exploration
   - Data augmentation (generates 900+ images from 6 originals)  
   - YOLO training with optimized settings
   - Results analysis and model testing
   - ONNX conversion for deployment

### **Method 3: Command Line (Advanced Users)**

1. **Generate augmented dataset:**
   ```bash
   python augment_dataset.py --input train/images/ --count 150
   ```

2. **Train the model:**
   ```bash
   python train_yolo.py --model n --epochs 100
   ```

3. **Convert to ONNX:**
   ```bash
   python convert_to_onnx.py --model runs/detect/drone_detection/weights/best.pt
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

## 📊 Expected Results

With 150 augmentations per class (900 total images), you should achieve:
- **99.5% mAP@0.5** accuracy on your 6 classes
- **Rotation invariance** up to 360°
- **Scale invariance** for drone altitude changes  
- **Good performance** in various lighting conditions
- **Fast inference**: 6-25ms on Mac M3, 2-5ms on RTX GPUs

## 📁 Key Files

- `YOLO_Training_Tutorial_Colab.ipynb` - **Google Colab notebook (EASIEST)**
- `YOLO_Training_Tutorial_Local.ipynb` - **Local Jupyter notebook**
- `augment_dataset.py` - Data augmentation script
- `train_yolo.py` - Command-line training script  
- `convert_to_onnx.py` - ONNX conversion for Pi deployment
- `dataset.yaml` - YOLO dataset configuration
- `requirements.txt` - Python dependencies

## 🥧 Raspberry Pi Deployment

The notebook includes automatic ONNX conversion optimized for Pi:
- **320x320 input** (perfect for Pi camera 320x240)
- **11.6MB model size** (lightweight for edge devices)
- **CPU optimized** (no GPU required)
- **Ready-to-use** inference examples

## 🛠️ Hardware Requirements

**Training:**
- **CPU**: Any modern processor (GPU highly recommended)
- **RAM**: 8GB+ recommended
- **GPU**: NVIDIA RTX series for fast training (optional but 10-20x faster)

**Inference:**
- **Pi 4**: 50-100ms per frame
- **Pi 5**: 30-60ms per frame  
- **Mac M3**: 6-25ms per frame
- **RTX 4070+**: 2-5ms per frame