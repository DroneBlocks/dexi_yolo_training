#!/usr/bin/env python3
"""
Interactive labeling tool for YOLO format labels
Click and drag to create bounding boxes, save labels

Usage:
    python3 label_images.py source_data/raw_drone_photos/cat      # auto-detects 'cat'
    python3 label_images.py source_data/raw_drone_photos/dog      # auto-detects 'dog'
    python3 label_images.py some/other/path --class bird          # specify class manually
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

class InteractiveLabelTool:
    def _load_classes(self):
        """Class names in training order.

        Walk up from the directory being labeled looking for a classes.txt, so
        each season names its own classes. Falls back to the 2025 list, which
        predates the file.
        """
        for d in [self.class_dir] + list(self.class_dir.parents):
            f = d / 'classes.txt'
            if f.exists():
                names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
                print(f"\U0001f4c4 Classes from {f}: {len(names)}")
                return names
        return ['car', 'motorcycle', 'truck', 'bird', 'cat', 'dog']

    def __init__(self, images_dir, default_class='dog'):
        self.class_dir = Path(images_dir)

        if not self.class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")

        # Standard YOLO structure: class/images/ and class/labels/
        self.images_dir = self.class_dir / 'images'
        self.labels_dir = self.class_dir / 'labels'

        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files in images/ subdirectory
        self.image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']:
            self.image_files.extend(sorted(self.images_dir.glob(ext)))

        if not self.image_files:
            raise FileNotFoundError(f"No images found in: {self.images_dir}\n" +
                                  f"Place images in {self.images_dir}/")

        self.current_idx = 0

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.labels = []
        self.mouse_pos = None  # Track mouse position for crosshair

        # Class settings. A season directory may carry a classes.txt listing its
        # classes in training order; without one, the 2025 list applies.
        self.class_names = self._load_classes()

        # Starting class: an explicit --class wins, then the directory name.
        # A wrong starting class is recoverable with the number keys; a silent
        # fallback is not, so an unrecognised --class is an error.
        dir_name = self.class_dir.name.lower()
        self.current_class = None

        if default_class:
            if default_class.lower() not in self.class_names:
                raise SystemExit(
                    "Unknown class %r. This set has: %s"
                    % (default_class, ', '.join(self.class_names)))
            self.current_class = self.class_names.index(default_class.lower())
            print(f"📝 Starting class from --class: {self.class_names[self.current_class]}")
        elif dir_name in self.class_names:
            self.current_class = self.class_names.index(dir_name)
            print(f"🔍 Auto-detected class from directory: {dir_name}")
        else:
            self.current_class = 0
            print(f"📝 Starting class: {self.class_names[0]} (press a number key to change)")

        print(f"\n{'='*70}")
        print(f"Interactive YOLO Labeler")
        print(f"{'='*70}")
        print(f"Directory: {self.images_dir}")
        print(f"Images found: {len(self.image_files)}")
        print(f"Labels will be saved to: {self.labels_dir}")
        print(f"\n{'='*70}")
        print("CONTROLS:")
        print(f"{'='*70}")
        print("  🖱️  Click and drag      - Draw bounding box")
        print("  n                     - Next image (auto-saves)")
        print("  p                     - Previous image (auto-saves)")
        print("  c                     - Clear all labels for current image")
        print("  u                     - Undo last box")
        print("  s                     - Save labels manually")
        print("  q                     - Quit (auto-saves)")
        print(f"  0-{len(self.class_names) - 1}                   - Change class:")
        for idx, name in enumerate(self.class_names):
            print(f"                          {idx}={name}")
        print(f"{'='*70}")
        print(f"Current class: {self.current_class} ({self.class_names[self.current_class]})")
        print(f"{'='*70}\n")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        # Always update mouse position for crosshair
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                end_point = (x, y)

                # Add the bounding box to labels
                self.add_label(self.start_point, end_point)
                self.current_rect = None

    def add_label(self, start_point, end_point):
        """Convert pixel coordinates to YOLO format and add label"""
        x1, y1 = start_point
        x2, y2 = end_point

        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Skip tiny boxes
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            print("⚠️  Box too small - skipped")
            return

        # Convert to YOLO format (normalized coordinates)
        img_h, img_w = self.current_image.shape[:2]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        # Add label (class_id, x_center, y_center, width, height)
        self.labels.append((self.current_class, x_center, y_center, width, height))
        self.touched = True
        print(f"✅ Added {self.class_names[self.current_class]} box (total: {len(self.labels)})")

    def load_labels(self, image_path):
        """Load existing YOLO format labels from labels/ directory"""
        label_path = self.labels_dir / (image_path.stem + '.txt')
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
            print(f"📂 Loaded {len(labels)} existing labels")

        return labels

    def save_labels(self, image_path):
        """Save labels to YOLO format file in labels/ directory"""
        label_path = self.labels_dir / (image_path.stem + '.txt')

        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in self.labels:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"💾 Saved {len(self.labels)} labels to {label_path.name}")

    def draw_labels(self, image):
        """Draw existing labels and current rectangle"""
        img = image.copy()
        h, w = img.shape[:2]

        # Draw existing labels
        for class_id, x_center, y_center, width, height in self.labels:
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)

            # One color per class. Ten entries so the 2026 set does not wrap and
            # give two classes the same box color.
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                      (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
                      (0, 128, 255), (128, 255, 0)]
            color = colors[class_id % len(colors)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw current rectangle being drawn
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw crosshair at mouse position (only when not drawing)
        if self.mouse_pos and not self.drawing:
            mx, my = self.mouse_pos
            # Draw dotted vertical line
            for y in range(0, h, 10):  # 10 pixel gaps for dotted effect
                cv2.line(img, (mx, y), (mx, min(y + 5, h)), (255, 255, 255), 1)
            # Draw dotted horizontal line
            for x in range(0, w, 10):  # 10 pixel gaps for dotted effect
                cv2.line(img, (x, my), (min(x + 5, w), my), (255, 255, 255), 1)

        return img

    def show_current_image(self):
        """Display current image with labels"""
        if self.current_idx >= len(self.image_files):
            return False

        image_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(image_path))

        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return True

        # Load existing labels for this image
        self.labels = self.load_labels(image_path)
        # Did this visit do anything? Quitting must not stamp an untouched frame
        # as an empty negative, which is what an empty label file asserts.
        self.touched = False

        # Create window and set mouse callback
        cv2.namedWindow('Interactive Labeler', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Interactive Labeler', self.mouse_callback)

        while True:
            # Draw image with labels
            display_img = self.draw_labels(self.current_image)

            # Add info text overlay
            info_text = f"Image {self.current_idx + 1}/{len(self.image_files)}: {image_path.name}"
            cv2.putText(display_img, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            class_text = f"Class: {self.current_class} ({self.class_names[self.current_class]})"
            cv2.putText(display_img, class_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            labels_text = f"Boxes: {len(self.labels)}"
            cv2.putText(display_img, labels_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow('Interactive Labeler', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # `n` means "done with this frame", so it always writes, empty or
                # not. `q` means "stop", which says nothing about the frame on
                # screen -- writing an empty file there would assert it holds
                # nothing when it was never looked at.
                if self.labels or self.touched:
                    self.save_labels(image_path)
                else:
                    print("↪️  Nothing drawn here, leaving it unlabeled")
                print("\n👋 Quitting...")
                return False
            elif key == ord('n'):
                self.save_labels(image_path)
                self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
                print(f"\n➡️  Next image ({self.current_idx + 1}/{len(self.image_files)})")
                break
            elif key == ord('p'):
                self.save_labels(image_path)
                self.current_idx = max(self.current_idx - 1, 0)
                print(f"\n⬅️  Previous image ({self.current_idx + 1}/{len(self.image_files)})")
                break
            elif key == ord('c'):
                self.labels.clear()
                self.touched = True
                print("🗑️  Cleared all labels for current image")
            elif key == ord('u'):
                if self.labels:
                    self.touched = True
                    removed = self.labels.pop()
                    print(f"↩️  Removed last label: {self.class_names[removed[0]]}")
                else:
                    print("⚠️  No labels to undo")
            elif key == ord('s'):
                self.touched = True
                self.save_labels(image_path)
            elif ord('0') <= key <= ord('9') and key - ord('0') < len(self.class_names):
                self.current_class = key - ord('0')
                print(f"🔄 Changed to class {self.current_class} ({self.class_names[self.current_class]})")

        return True

    def run(self):
        """Main loop"""
        while self.current_idx < len(self.image_files):
            if not self.show_current_image():
                break

        cv2.destroyAllWindows()

        print(f"\n{'='*70}")
        print("✅ Labeling session complete!")
        print(f"{'='*70}")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")
        done = len(list(self.labels_dir.glob('*.txt')))
        print(f"\nLabeled {done} of {len(self.image_files)} images.")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Interactive YOLO labeling tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect class from directory name:
  python3 label_images.py source_data/raw_drone_photos/cat
  python3 label_images.py source_data/raw_drone_photos/dog
  python3 label_images.py source_data/raw_drone_photos/bird

  # Or specify class manually with --class:
  python3 label_images.py some/other/path --class bird
  python3 label_images.py my_images --class motorcycle

Controls:
  Click and drag to draw bounding box
  'n' - Next image    'p' - Previous image
  'c' - Clear labels  'u' - Undo last box
  's' - Save          'q' - Quit
  '0-5' - Change class (0=car, 1=motorcycle, 2=truck, 3=bird, 4=cat, 5=dog)
        """
    )
    parser.add_argument('directory', type=str,
                       help='Directory containing images to label')
    parser.add_argument('--class', '-c', dest='default_class', type=str, default=None,
                       help='Starting class (validated against the set\'s classes.txt)')

    args = parser.parse_args()

    try:
        labeler = InteractiveLabelTool(args.directory, args.default_class)
        labeler.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nUsage: python3 label_images.py <class_directory>")
        print(f"\n⚠️  Important: Provide the CLASS directory, not the images/ subdirectory")
        print(f"\nCorrect examples:")
        print(f"  ✅ python3 label_images.py source_data/raw_drone_photos/cat")
        print(f"  ✅ python3 label_images.py source_data/raw_drone_photos/bird")
        print(f"\nIncorrect examples:")
        print(f"  ❌ python3 label_images.py source_data/raw_drone_photos/cat/images")
        print(f"  ❌ python3 label_images.py source_data/raw_drone_photos/bird/images")
        print(f"\nThe script automatically looks for images/ and labels/ inside the directory you provide.\n")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
