#!/usr/bin/env python3
"""
Convert YOLO PyTorch model to ONNX format for Raspberry Pi deployment.
Optimizes the model for better performance on edge devices.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch

def convert_model_to_onnx(model_path, output_dir=None, imgsz=640, half=False, simplify=True):
    """
    Convert YOLO PyTorch model to ONNX format
    
    Args:
        model_path: Path to the .pt model file
        output_dir: Output directory for ONNX file (default: same as model)
        imgsz: Image size for export (default: 640)
        half: Use FP16 precision (default: False - better for Pi)
        simplify: Simplify ONNX model (default: True)
    """
    
    # Validate input file
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"🚀 Converting YOLO model to ONNX...")
    print(f"📂 Input model: {model_path}")
    print(f"🖼️  Image size: {imgsz}x{imgsz}")
    print(f"🔧 Half precision: {half}")
    print(f"⚡ Simplify: {simplify}")
    
    # Load the model
    print(f"\n📥 Loading model...")
    model = YOLO(str(model_path))
    
    # Print model info
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"   Size on disk: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Set output directory
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_name = model_path.stem + "_optimized.onnx"
    output_path = output_dir / output_name
    
    print(f"\n🔄 Converting to ONNX...")
    print(f"📁 Output path: {output_path}")
    
    try:
        # Export to ONNX
        success = model.export(
            format='onnx',
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            workspace=4,  # TensorRT workspace size in GB
            verbose=True
        )
        
        # Find the generated ONNX file (YOLO auto-generates the name)
        generated_onnx = model_path.parent / (model_path.stem + ".onnx")
        
        if generated_onnx.exists():
            # Rename to our desired name
            if generated_onnx != output_path:
                if output_path.exists():
                    output_path.unlink()  # Remove existing file
                generated_onnx.rename(output_path)
            
            print(f"\n🎉 Conversion successful!")
            print(f"✅ ONNX model saved: {output_path}")
            print(f"📦 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Verify the ONNX model
            verify_onnx_model(output_path, model)
            
            # Print deployment instructions
            print_deployment_instructions(output_path)
            
            return str(output_path)
            
        else:
            print("❌ ONNX file not found after export")
            return None
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def verify_onnx_model(onnx_path, original_model):
    """Verify the ONNX model can be loaded and used"""
    try:
        print(f"\n🔍 Verifying ONNX model...")
        
        # Try to load with ONNX Runtime (if available)
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            print(f"✅ ONNX model verification successful!")
            print(f"   Input shape: {inputs[0].shape}")
            print(f"   Input type: {inputs[0].type}")
            print(f"   Output shape: {outputs[0].shape}")
            print(f"   Providers: {ort.get_available_providers()}")
            
        except ImportError:
            print("⚠️  ONNX Runtime not available for verification")
            print("   Install with: pip install onnxruntime")
            
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")

def print_deployment_instructions(onnx_path):
    """Print instructions for Pi deployment"""
    print(f"\n" + "="*60)
    print(f"🥧 RASPBERRY PI DEPLOYMENT INSTRUCTIONS")
    print(f"="*60)
    print(f"""
📋 Steps to deploy on Raspberry Pi:

1. 📁 Copy files to Pi:
   scp {onnx_path} pi@your-pi-ip:~/
   scp dataset.yaml pi@your-pi-ip:~/

2. 🔧 Install dependencies on Pi:
   pip install onnxruntime opencv-python numpy
   # OR for better performance:
   pip install onnxruntime-gpu opencv-python numpy

3. 🐍 Python inference example:
   ```python
   import onnxruntime as ort
   import cv2
   import numpy as np
   
   # Load ONNX model
   session = ort.InferenceSession('{onnx_path.name}')
   
   # Load and preprocess image
   img = cv2.imread('test_image.jpg')
   img = cv2.resize(img, (640, 640))
   img = img.astype(np.float32) / 255.0
   img = np.transpose(img, (2, 0, 1))
   img = np.expand_dims(img, 0)
   
   # Run inference
   outputs = session.run(None, {{'images': img}})
   print("Detections:", outputs[0])
   ```

4. ⚡ For even better Pi performance:
   - Use lower image size: 416x416 or 320x320
   - Consider quantizing to INT8
   - Use TensorRT if available (Jetson Nano)

5. 🎯 Class mapping:
   0: bird, 1: dog, 2: cat, 3: motorcycle, 4: car, 5: truck
""")
    print(f"="*60)

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO PyTorch model to ONNX')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to PyTorch model (.pt file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: same as model)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for export (default: 640)')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 precision (may not work on all Pi models)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Do not simplify ONNX model')
    
    args = parser.parse_args()
    
    # Convert model
    onnx_path = convert_model_to_onnx(
        model_path=args.model,
        output_dir=args.output,
        imgsz=args.imgsz,
        half=args.half,
        simplify=not args.no_simplify
    )
    
    if onnx_path:
        print(f"\n🎉 Success! ONNX model ready for Pi deployment: {onnx_path}")
    else:
        print(f"\n❌ Conversion failed!")
        exit(1)

if __name__ == "__main__":
    main()