#!/usr/bin/env python3
"""
Data augmentation script for YOLO training dataset.
Generates rotated, scaled, and transformed variants of original images.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

class YOLODatasetAugmenter:
    def __init__(self, original_images_dir="source_data/original_images", output_dir="train", val_split=0.2):
        self.original_images_dir = Path(original_images_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split

        # Create train directories
        self.train_images_dir = self.output_dir / "images"
        self.train_labels_dir = self.output_dir / "labels"

        # Create validation directories
        self.val_images_dir = Path("val") / "images"
        self.val_labels_dir = Path("val") / "labels"

        # Create all output directories
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)

        # Class mapping: Sequential IDs (0-5) but ordered for better COCO transfer learning
        self.class_to_id = {
            'car': 0,        # Sequential ID 0 (maps to COCO ID 2)
            'motorcycle': 1, # Sequential ID 1 (maps to COCO ID 3)
            'truck': 2,      # Sequential ID 2 (maps to COCO ID 7)
            'bird': 3,       # Sequential ID 3 (maps to COCO ID 14)
            'cat': 4,        # Sequential ID 4 (maps to COCO ID 15)
            'dog': 5,        # Sequential ID 5 (maps to COCO ID 16)
        }
        self.class_names = list(self.class_to_id.keys())

    def load_labels(self, label_path):
        """Load YOLO format labels from file"""
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, x_center, y_center, width, height))
        return labels
        
    def detect_class_from_filename(self, filename):
        """Detect class from filename"""
        filename_lower = filename.lower()
        for class_name in self.class_names:
            if class_name in filename_lower:
                return self.class_to_id[class_name]
        raise ValueError(f"Could not detect class from filename: {filename}")
    
    def rotate_image_and_bbox(self, image, bboxes, angle):
        """Rotate image and transform bounding boxes

        Args:
            image: Input image
            bboxes: List of (class_id, x_center, y_center, width, height) in normalized coords
            angle: Rotation angle in degrees

        Returns:
            rotated_image, rotated_bboxes
        """
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

        # Transform bounding boxes
        rotated_bboxes = []
        for class_id, x_center, y_center, width_norm, height_norm in bboxes:
            # Convert normalized coordinates to pixel coordinates
            x_center_px = x_center * w
            y_center_px = y_center * h
            width_px = width_norm * w
            height_px = height_norm * h

            # Get the four corners of the bounding box
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2

            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])

            # Transform corners using rotation matrix
            ones = np.ones(shape=(len(corners), 1))
            corners_with_ones = np.hstack([corners, ones])
            transformed_corners = M.dot(corners_with_ones.T).T

            # Get bounding box of rotated corners
            x_coords = transformed_corners[:, 0]
            y_coords = transformed_corners[:, 1]

            new_x1 = np.min(x_coords)
            new_y1 = np.min(y_coords)
            new_x2 = np.max(x_coords)
            new_y2 = np.max(y_coords)

            # Clip to image boundaries
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(new_w, new_x2)
            new_y2 = min(new_h, new_y2)

            # Convert back to normalized YOLO format
            new_x_center = ((new_x1 + new_x2) / 2) / new_w
            new_y_center = ((new_y1 + new_y2) / 2) / new_h
            new_width = (new_x2 - new_x1) / new_w
            new_height = (new_y2 - new_y1) / new_h

            # Only keep boxes that are still visible
            if new_width > 0.01 and new_height > 0.01:
                rotated_bboxes.append((class_id, new_x_center, new_y_center, new_width, new_height))

        return rotated, rotated_bboxes
    
    def scale_image_and_bbox(self, image, bboxes, scale_factor):
        """Scale image and transform bounding boxes

        Args:
            image: Input image
            bboxes: List of (class_id, x_center, y_center, width, height) in normalized coords
            scale_factor: Scale factor (>1.0 zooms in, <1.0 zooms out)

        Returns:
            scaled_image, scaled_bboxes
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        scaled_bboxes = []

        if scale_factor > 1.0:
            # Scale up then crop to original size (zoom in)
            scaled = cv2.resize(image, (new_w, new_h))
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            cropped = scaled[start_y:start_y+h, start_x:start_x+w]

            # Transform bboxes: scale up then adjust for crop
            for class_id, x_center, y_center, width_norm, height_norm in bboxes:
                # Scale up coordinates
                new_x_center = x_center * scale_factor
                new_y_center = y_center * scale_factor
                new_width = width_norm * scale_factor
                new_height = height_norm * scale_factor

                # Adjust for crop offset
                crop_offset_x = start_x / new_w
                crop_offset_y = start_y / new_h

                new_x_center = (new_x_center * new_w - start_x) / w
                new_y_center = (new_y_center * new_h - start_y) / h
                new_width = new_width * new_w / w
                new_height = new_height * new_h / h

                # Check if bbox is still in bounds
                if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                    new_width > 0.01 and new_height > 0.01):
                    # Clip to bounds
                    new_x_center = max(0, min(1, new_x_center))
                    new_y_center = max(0, min(1, new_y_center))
                    scaled_bboxes.append((class_id, new_x_center, new_y_center, new_width, new_height))

            return cropped, scaled_bboxes
        else:
            # Scale down then pad to original size (zoom out)
            scaled = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = scaled

            # Transform bboxes: scale down then adjust for padding
            for class_id, x_center, y_center, width_norm, height_norm in bboxes:
                # Scale down and add padding offset
                new_x_center = (x_center * new_w + start_x) / w
                new_y_center = (y_center * new_h + start_y) / h
                new_width = width_norm * scale_factor
                new_height = height_norm * scale_factor

                scaled_bboxes.append((class_id, new_x_center, new_y_center, new_width, new_height))

            return canvas, scaled_bboxes
    
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
    
    def generate_augmentations(self, image_path, label_path, class_name, augmentations_per_image=100):
        """Generate augmented versions of a single image with proper bbox transformations

        Args:
            image_path: Path to the image file
            label_path: Path to the label file
            class_name: Name of the class
            augmentations_per_image: Number of augmentations to generate
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Load labels
        labels = self.load_labels(label_path)
        if not labels:
            print(f"Warning: No labels found for {image_path.name}, skipping")
            return

        # Calculate train/val split
        val_count = int(augmentations_per_image * self.val_split)
        train_count = augmentations_per_image - val_count

        print(f"Processing {class_name} - generating {augmentations_per_image} variations")
        print(f"  Train: {train_count}, Validation: {val_count}")
        print(f"  Original labels: {len(labels)} bounding box(es)")

        # Generate augmentations
        for i in range(augmentations_per_image):
            # Start with original image and labels
            aug_image = image.copy()
            aug_labels = labels.copy()

            # Random rotation (0-360 degrees)
            angle = np.random.uniform(0, 360)
            aug_image, aug_labels = self.rotate_image_and_bbox(aug_image, aug_labels, angle)

            # Skip if all labels were lost during rotation
            if not aug_labels:
                continue

            # Random scale (0.25x to 1.3x)
            scale = np.random.uniform(0.25, 1.3)
            aug_image, aug_labels = self.scale_image_and_bbox(aug_image, aug_labels, scale)

            # Skip if all labels were lost during scaling
            if not aug_labels:
                continue

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

            # Determine if this image goes to train or validation
            if i < train_count:
                images_dir = self.train_images_dir
                labels_dir = self.train_labels_dir
            else:
                images_dir = self.val_images_dir
                labels_dir = self.val_labels_dir

            # Save augmented image
            img_filename = f"{class_name}_{i+1:03d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), aug_image)

            # Save corresponding label file with all bounding boxes
            label_filename = f"{class_name}_{i+1:03d}.txt"
            label_path = labels_dir / label_filename

            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            with open(label_path, 'w') as f:
                for class_id, x_center, y_center, width, height in aug_labels:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_all_images(self, augmentations_per_image=100):
        """Process all images in class subdirectories with YOLO structure

        Expected structure:
            original_images/
                <class>/
                    images/
                        <class>.jpg
                    labels/
                        <class>.txt
        """
        print(f"\nScanning for images in {self.original_images_dir}/")
        print(f"Expected structure: <class>/images/ and <class>/labels/\n")

        # Find all class directories
        class_dirs = [d for d in self.original_images_dir.iterdir() if d.is_dir() and d.name in self.class_names]

        if not class_dirs:
            print(f"No class directories found in {self.original_images_dir}")
            print(f"Expected directories: {', '.join(self.class_names)}")
            return

        total_images = 0

        for class_dir in class_dirs:
            class_name = class_dir.name
            images_dir = class_dir / 'images'
            labels_dir = class_dir / 'labels'

            if not images_dir.exists():
                print(f"Warning: {class_name}/images/ not found, skipping")
                continue

            # Find all images in this class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(images_dir.glob(ext))
                image_files.extend(images_dir.glob(ext.upper()))

            if not image_files:
                print(f"Warning: No images found in {class_name}/images/, skipping")
                continue

            print(f"Found {len(image_files)} image(s) for class '{class_name}'")

            # Process each image
            for image_path in image_files:
                label_path = labels_dir / f"{image_path.stem}.txt"
                self.generate_augmentations(image_path, label_path, class_name, augmentations_per_image)
                total_images += 1

        print(f"\n{'='*60}")
        print(f"Augmentation complete!")
        print(f"{'='*60}")
        print(f"Processed {total_images} original images")
        print(f"Training images: {len(list(self.train_images_dir.glob('*.jpg')))}")
        print(f"Training labels: {len(list(self.train_labels_dir.glob('*.txt')))}")
        print(f"Validation images: {len(list(self.val_images_dir.glob('*.jpg')))}")
        print(f"Validation labels: {len(list(self.val_labels_dir.glob('*.txt')))}")
        print(f"Total augmented images: {len(list(self.train_images_dir.glob('*.jpg'))) + len(list(self.val_images_dir.glob('*.jpg')))}")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Augment YOLO training dataset')
    parser.add_argument('--input', '-i', type=str, default="source_data/original_images",
                       help='Directory containing original images (default: source_data/original_images)')
    parser.add_argument('--output', '-o', type=str, default='train',
                       help='Output directory for augmented dataset')
    parser.add_argument('--count', '-c', type=int, default=100,
                       help='Number of augmentations per original image')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data to use for validation (default: 0.2)')

    args = parser.parse_args()

    input_dir = args.input if args.input else "source_data/original_images"
    augmenter = YOLODatasetAugmenter(input_dir, args.output, args.val_split)
    augmenter.augment_all_images(args.count)

if __name__ == "__main__":
    main()