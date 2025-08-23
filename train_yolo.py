#!/usr/bin/env python3
"""
YOLO training script for custom 6-class drone detection dataset.
"""

from ultralytics import YOLO
import argparse

def train_yolo_model(model_size='n', epochs=100, imgsz=640, batch_size=16, device='cpu'):
    """
    Train a YOLO model on the custom dataset
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        device: Device to use ('cpu', 'cuda', 'mps')
    """
    
    # Load a pre-trained YOLO model
    model = YOLO(f'yolov8{model_size}.pt')
    
    print(f"Training YOLOv8{model_size} model...")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project='runs/detect',
        name='drone_detection',
        save_period=10,  # Save checkpoint every 10 epochs
        patience=20,     # Early stopping patience
        
        # Augmentation settings (additional to our pre-generated augmentations)
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation  
        hsv_v=0.4,       # Value augmentation
        degrees=0,       # Don't add rotation (we already did this)
        translate=0.1,   # Translation augmentation
        scale=0.1,       # Additional scale augmentation
        shear=0.1,       # Shear augmentation
        perspective=0.0, # Perspective augmentation
        flipud=0.0,      # No vertical flip (objects have orientation)
        fliplr=0.0,      # No horizontal flip (for consistency)
        mosaic=0.8,      # Mosaic augmentation probability
        mixup=0.1,       # Mixup augmentation probability
        
        # Optimization
        optimizer='AdamW',
        lr0=0.01,        # Initial learning rate
        lrf=0.1,         # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Other settings
        box=7.5,         # Box loss gain
        cls=0.5,         # Class loss gain
        dfl=1.5,         # DFL loss gain
        verbose=True,
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: runs/detect/drone_detection/weights/best.pt")
    print(f"Last model saved at: runs/detect/drone_detection/weights/last.pt")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    
    return model, results, metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for drone detection')
    parser.add_argument('--model', '-m', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    model, results, metrics = train_yolo_model(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )

if __name__ == "__main__":
    main()