#!/usr/bin/env python3
"""
View augmented dataset images with bounding boxes overlaid.
Navigate through train/ or val/ directories to verify augmentation quality.

Usage:
    python3 view_augmented.py train    # View training set
    python3 view_augmented.py val      # View validation set
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

class AugmentedDatasetViewer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Class mapping
        self.id_to_class = {
            0: 'car',
            1: 'motorcycle',
            2: 'truck',
            3: 'bird',
            4: 'cat',
            5: 'dog',
        }

        # Colors for each class (BGR format)
        self.class_colors = {
            0: (0, 255, 255),    # Yellow - car
            1: (255, 0, 255),    # Magenta - motorcycle
            2: (255, 128, 0),    # Orange - truck
            3: (0, 255, 0),      # Green - bird
            4: (255, 0, 0),      # Blue - cat
            5: (0, 0, 255),      # Red - dog
        }

        # Find all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(self.images_dir.glob(ext))
            self.image_files.extend(self.images_dir.glob(ext.upper()))

        self.image_files = sorted(self.image_files)

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.current_idx = 0

    def load_labels(self, image_path):
        """Load YOLO format labels for an image"""
        label_path = self.labels_dir / f"{image_path.stem}.txt"
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

    def draw_labels_on_image(self, image, labels):
        """Draw bounding boxes and labels on image"""
        img_height, img_width = image.shape[:2]
        display_image = image.copy()

        for class_id, x_center, y_center, width_norm, height_norm in labels:
            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * img_width)
            y_center_px = int(y_center * img_height)
            width_px = int(width_norm * img_width)
            height_px = int(height_norm * img_height)

            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # Get class name and color
            class_name = self.id_to_class.get(class_id, f'Class_{class_id}')
            color = self.class_colors.get(class_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f'{class_name}'
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            label_y = max(y1 - 10, text_height + 10)
            cv2.rectangle(
                display_image,
                (x1, label_y - text_height - baseline - 5),
                (x1 + text_width + 5, label_y + baseline),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                display_image,
                label_text,
                (x1 + 2, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # Draw box info (normalized size)
            info_text = f'w:{width_norm:.3f} h:{height_norm:.3f}'
            cv2.putText(
                display_image,
                info_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        return display_image

    def view_image(self, image_path):
        """Display a single image with labels"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return True

        # Load labels
        labels = self.load_labels(image_path)

        # Draw labels on image
        display_image = self.draw_labels_on_image(image, labels)

        # Add navigation info overlay
        info_y = 30
        cv2.putText(display_image, f"Image {self.current_idx + 1}/{len(self.image_files)}: {image_path.name}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_image, f"Boxes: {len(labels)}",
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_image, "Controls: [n]ext [p]rev [q]uit [f]irst [l]ast",
                   (10, display_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display info in console
        print(f"\n[{self.current_idx + 1}/{len(self.image_files)}] {image_path.name}")
        print(f"  Bounding boxes: {len(labels)}")
        for class_id, x_center, y_center, width, height in labels:
            class_name = self.id_to_class.get(class_id, f'Class_{class_id}')
            print(f"    - {class_name}: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}×{height:.3f})")

        # Create window and display
        window_name = f'Augmented Dataset Viewer - {self.dataset_dir.name}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Resize window to fit screen better
        img_height, img_width = image.shape[:2]
        max_display_height = 800
        if img_height > max_display_height:
            scale = max_display_height / img_height
            display_width = int(img_width * scale)
            cv2.resizeWindow(window_name, display_width, max_display_height)

        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)

        # Handle key presses
        if key == 27 or key == ord('q'):  # ESC or 'q'
            return False
        elif key == ord('n'):  # Next
            self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
        elif key == ord('p'):  # Previous
            self.current_idx = max(self.current_idx - 1, 0)
        elif key == ord('f'):  # First
            self.current_idx = 0
        elif key == ord('l'):  # Last
            self.current_idx = len(self.image_files) - 1

        return True

    def run(self):
        """Main viewer loop"""
        print(f"\n{'='*70}")
        print(f"Augmented Dataset Viewer: {self.dataset_dir}")
        print(f"{'='*70}")
        print(f"Images: {len(self.image_files)}")
        print(f"Location: {self.images_dir}")
        print(f"\nControls:")
        print(f"  n - Next image")
        print(f"  p - Previous image")
        print(f"  f - First image")
        print(f"  l - Last image")
        print(f"  q or ESC - Quit")
        print(f"{'='*70}")

        while True:
            image_path = self.image_files[self.current_idx]
            should_continue = self.view_image(image_path)
            if not should_continue:
                break

        print(f"\n{'='*70}")
        print("Done viewing images!")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='View augmented dataset images with bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 view_augmented.py train    # View training set
  python3 view_augmented.py val      # View validation set
        """
    )
    parser.add_argument('dataset', type=str, choices=['train', 'val'],
                       help='Dataset to view (train or val)')

    args = parser.parse_args()

    try:
        viewer = AugmentedDatasetViewer(args.dataset)
        viewer.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure you've run augmentation first:")
        print(f"  python3 augment_dataset.py\n")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
