# 🌟 Google Colab Setup Guide

## Quick Start (2 minutes)

### 1. **Open in Google Colab**
- Go to [colab.research.google.com](https://colab.research.google.com/)
- Click "File" → "Upload notebook" 
- Upload `YOLO_Training_Tutorial.ipynb`

**OR** use this direct link (replace with your GitHub repo):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/YOLO_Training_Tutorial.ipynb)

### 2. **Enable GPU (Critical for Speed)**
- In Colab: `Runtime` → `Change runtime type` → `Hardware accelerator` → `T4 GPU`
- You should see: "🚀 GPU detected: Tesla T4" in the first cell

### 3. **Upload Your Files**
Two options:

**Option A: Drag & Drop (Easiest)**
- Click the 📁 folder icon on the left sidebar
- Create folders: `train/images/`  
- Drag your 6 base images into `train/images/`
- Upload `dataset.yaml` and `augment_dataset.py` to root

**Option B: Use Upload Widget**
- The notebook includes an upload widget
- Follow the prompts to upload files

### 4. **Run the Notebook**
- Click "Runtime" → "Run all" 
- Or run cells one by one with Shift+Enter

## 📊 Expected Performance

**With Tesla T4 GPU (Free Tier):**
- **Data augmentation**: 2-3 minutes (900 images)  
- **Training 100 epochs**: 15-25 minutes
- **Final mAP@0.5**: 99.5%+
- **ONNX conversion**: 30 seconds

**Compare to local CPU:** 10-20x faster!

## 💾 Download Your Models

After training completes:
1. Navigate to `runs/detect/drone_detection/weights/`
2. Right-click `best.pt` → "Download"
3. Right-click `best_optimized.onnx` → "Download"

Your trained models are ready for Pi deployment!

## ⚠️ Colab Limitations & Tips

### **Session Limits:**
- **Free tier**: 12 hours max per session
- **Solution**: Training completes in ~30 minutes (well within limit)

### **File Persistence:**
- Files are deleted when session ends
- **Solution**: Download models immediately after training

### **Memory:**
- 12GB RAM available (plenty for this project)
- If you get OOM errors, reduce batch size to 8

### **GPU Time Limits:**
- Free tier has daily GPU quota
- **Solution**: Our training needs <1 hour of GPU time

## 🚀 Pro Tips for Colab

### **Faster Startup:**
```python
# Run this first to cache common downloads
!pip install ultralytics -q
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Pre-downloads base model
```

### **Monitor Training:**
```python
# Real-time training monitoring
from IPython.display import clear_output
import time

# The notebook includes built-in monitoring
```

### **Save to Google Drive (Optional):**
```python
# Mount Google Drive to save models permanently  
from google.colab import drive
drive.mount('/content/drive')

# Copy models to Drive after training
!cp runs/detect/*/weights/best.pt /content/drive/MyDrive/
```

## 🔧 Troubleshooting

### **"No module named 'augment_dataset'"**
- Make sure you uploaded `augment_dataset.py` to the root directory
- Check the file browser on the left - it should show the file

### **"Dataset files not found"**
- Upload your 6 base images to `train/images/` folder
- Upload `dataset.yaml` to root directory
- Make sure filenames match: `bird.jpg`, `car.jpg`, etc.

### **GPU not working**
- Go to Runtime → Change runtime type → GPU
- Restart runtime: Runtime → Restart runtime
- Check with: `torch.cuda.is_available()` (should return True)

### **Training very slow**
- Confirm GPU is enabled (see above)
- Reduce batch size: change `'batch_size': 8` in cell 14
- Reduce epochs: change `'epochs': 50` for faster training

## 📱 Mobile Access

Colab works great on tablets/phones:
- Upload files from your phone's camera roll  
- Monitor training progress on mobile
- Download trained models to phone

Perfect for field work! 📸🚁

---

**Need help?** The notebook includes detailed error handling and troubleshooting steps.