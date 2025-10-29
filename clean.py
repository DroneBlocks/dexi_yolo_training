#!/usr/bin/env python3
"""
Clean all training artifacts to start fresh.
Removes augmented datasets, training results, and model outputs.
"""

import shutil
from pathlib import Path
import sys

def clean_training_data():
    """Remove all training-related directories and files"""

    # Directories to remove
    dirs_to_remove = [
        'train',           # Augmented training data
        'val',             # Augmented validation data
        'combined',        # Combined dataset (augmented + real)
        'results',         # Training results and metrics
        'runs',            # YOLO training runs (alternative location)
    ]

    # Individual files to remove (trained models in root)
    files_to_remove = [
        'best.pt',
        'last.pt',
        'yolov8n.pt',      # Downloaded pretrained weights
        'yolov8s.pt',
        'yolov8m.pt',
        'yolov8l.pt',
        'yolov8x.pt',
    ]

    removed_count = 0

    print("\n" + "="*70)
    print("🧹 Cleaning Training Artifacts")
    print("="*70)
    print("\nThis will remove:")
    print("  - Augmented training/validation data (train/, val/)")
    print("  - Combined datasets (combined/)")
    print("  - Training results (results/, runs/)")
    print("  - Trained model files (*.pt)")
    print("\nOriginal source data will NOT be touched:")
    print("  ✓ source_data/original_images/")
    print("  ✓ source_data/real_drone_photos/")
    print("="*70)

    response = input("\nProceed with cleaning? [y/N]: ").strip().lower()

    if response != 'y':
        print("\n❌ Cleaning cancelled")
        return

    print("\n🚮 Removing directories...")
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed: {dir_name}/")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing {dir_name}: {e}")
        else:
            print(f"  - Skipped: {dir_name}/ (doesn't exist)")

    print("\n🚮 Removing model files...")
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  ✓ Removed: {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing {file_name}: {e}")

    print("\n" + "="*70)
    if removed_count > 0:
        print(f"✅ Cleaning complete! Removed {removed_count} items")
        print("\nYou can now start fresh:")
        print("  1. python augment_dataset.py --count 100")
        print("  2. python train_with_real_data.py")
    else:
        print("✅ Nothing to clean - workspace is already clean!")
    print("="*70 + "\n")

def main():
    try:
        clean_training_data()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cleaning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
