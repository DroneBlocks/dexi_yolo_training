#!/usr/bin/env python3
"""
Data augmentation script for YOLO training dataset.
Generates rotated, scaled, and transformed variants of base images.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

class YOLODatasetAugmenter:
    def __init__(self, base_images_dir, output_dir="train"):
        self.base_images_dir = Path(base_images_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping for your 6 classes
        self.class_names = ['bird', 'dog', 'cat', 'motorcycle', 'car', 'truck']
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
    def detect_class_from_filename(self, filename):
        """Detect class from filename"""
        filename_lower = filename.lower()
        for class_name in self.class_names:
            if class_name in filename_lower:
                return self.class_to_id[class_name]
        raise ValueError(f"Could not detect class from filename: {filename}")
    
    def rotate_image_and_bbox(self, image, angle):
        """Rotate image and return full bounding box for rotated object"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new canvas size to fit rotated image
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))
        
        # Adjust rotation matrix for new canvas size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(0, 0, 0))
        
        # For single object detection, bbox covers entire image with some padding
        padding = 0.05  # 5% padding
        x_center = 0.5
        y_center = 0.5
        width = 1.0 - 2 * padding
        height = 1.0 - 2 * padding
        
        return rotated, (x_center, y_center, width, height)
    
    def scale_image(self, image, scale_factor):
        """Scale image up or down"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        if scale_factor > 1.0:
            # Scale up then crop to original size
            scaled = cv2.resize(image, (new_w, new_h))
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            cropped = scaled[start_y:start_y+h, start_x:start_x+w]
            return cropped
        else:
            # Scale down then pad to original size
            scaled = cv2.resize(image, (new_w, new_h))
            # Create black canvas of original size
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
            return canvas
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Adjust image brightness and contrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def add_noise(self, image, noise_factor=25):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def blur_image(self, image, blur_strength=3):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    def generate_augmentations(self, image_path, augmentations_per_image=100):
        """Generate augmented versions of a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Detect class from filename
        class_id = self.detect_class_from_filename(image_path.name)
        class_name = self.class_names[class_id]
        
        print(f"Processing {class_name} - generating {augmentations_per_image} variations")
        
        # Generate augmentations
        for i in range(augmentations_per_image):
            # Start with original image
            aug_image = image.copy()
            
            # Random rotation (0-360 degrees)
            angle = np.random.uniform(0, 360)
            aug_image, bbox = self.rotate_image_and_bbox(aug_image, angle)
            
            # Random scale (0.7x to 1.3x)
            scale = np.random.uniform(0.7, 1.3)
            aug_image = self.scale_image(aug_image, scale)
            
            # Random brightness (-30 to +30)
            brightness = np.random.randint(-30, 31)
            # Random contrast (0.7 to 1.3)
            contrast = np.random.uniform(0.7, 1.3)
            aug_image = self.adjust_brightness_contrast(aug_image, brightness, contrast)
            
            # Random noise (20% chance)
            if np.random.random() < 0.2:
                aug_image = self.add_noise(aug_image)
            
            # Random blur (15% chance)
            if np.random.random() < 0.15:
                blur_strength = np.random.choice([3, 5, 7])
                aug_image = self.blur_image(aug_image, blur_strength)
            
            # Save augmented image
            img_filename = f"{class_name}_{i+1:03d}.jpg"
            img_path = self.images_dir / img_filename
            cv2.imwrite(str(img_path), aug_image)
            
            # Save corresponding label file
            label_filename = f"{class_name}_{i+1:03d}.txt"
            label_path = self.labels_dir / label_filename
            
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def augment_all_images(self, augmentations_per_image=100):
        """Process all images in the base directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(self.base_images_dir.glob(ext))
            image_files.extend(self.base_images_dir.glob(ext.upper()))
        
        if not image_files:
            print(f"No image files found in {self.base_images_dir}")
            return
        
        print(f"Found {len(image_files)} base images")
        
        for image_path in image_files:
            self.generate_augmentations(image_path, augmentations_per_image)
        
        print(f"\nAugmentation complete!")
        print(f"Generated images: {len(list(self.images_dir.glob('*.jpg')))}")
        print(f"Generated labels: {len(list(self.labels_dir.glob('*.txt')))}")

def main():
    parser = argparse.ArgumentParser(description='Augment YOLO training dataset')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Directory containing base images')
    parser.add_argument('--output', '-o', type=str, default='train',
                       help='Output directory for augmented dataset')
    parser.add_argument('--count', '-c', type=int, default=100,
                       help='Number of augmentations per base image')
    
    args = parser.parse_args()
    
    augmenter = YOLODatasetAugmenter(args.input, args.output)
    augmenter.augment_all_images(args.count)

if __name__ == "__main__":
    main()