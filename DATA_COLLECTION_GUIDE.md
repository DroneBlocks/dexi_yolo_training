# Drone Data Collection Guide

## Objective
Collect real-world drone camera images of the 6 COCO classes (bird, car, cat, dog, motorcycle, truck) to fine-tune YOLOv8 for drone-based detection in recessed buildings.

## Classes to Photograph
- bird
- car
- cat
- dog
- motorcycle
- truck

## Collection Protocol

### Setup
1. Print images of the 6 classes (bird, car, cat, dog, motorcycle, truck)
2. Place images in the recessed building setup (actual deployment scenario)
3. Ensure drone camera is pointing straight down (matches production)

### Photo Requirements Per Class

**Target: 40-60 photos per class**

#### Drone Approach Angles (CRITICAL for recessed buildings)
Rotate drone around building to capture how shadows/occlusion change:
- **North approach** (0°)
- **East approach** (90°)
- **South approach** (180°)
- **West approach** (270°)

WHY: Building shadows and edge occlusion change with approach angle - this is domain-specific data that augmentation cannot replicate!

#### Height Variations (per approach angle)
- 0.5 meters above court
- 1.0 meters above court
- 2.0 meters above court
- 3.0 meters above court
- 4.0 meters above court

#### Optimal Collection Matrix
For efficiency, collect at 2-3 heights per approach angle:
- 4 angles × 3 heights = 12 photos per class (minimum)
- 4 angles × 5 heights = 20 photos per class (recommended)

#### Additional Variations
- Lighting conditions (time of day, indoor/outdoor)
- Position in frame (center, off-center)
- Partial occlusion (edge of building)
- Distance variations within each height

### Naming Convention
```
<class>_drone_<height>m_<angle>deg_<condition>_<number>.jpg
```

Examples:
- `bird_drone_1m_0deg_shadow_001.jpg` (North approach, 1m height)
- `car_drone_3m_90deg_indoor_001.jpg` (East approach, 3m height)
- `dog_drone_2m_180deg_natural_001.jpg` (South approach, 2m height)
- `cat_drone_4m_270deg_shadow_001.jpg` (West approach, 4m height)

### Storage
Save all collected images to:
```
source_data/real_drone_photos/
  ├── bird/
  │   ├── images/          # Put your .jpg files here
  │   └── labels/          # Labels will be created here
  ├── car/
  │   ├── images/
  │   └── labels/
  ├── cat/
  │   ├── images/
  │   └── labels/
  ├── dog/
  │   ├── images/
  │   └── labels/
  ├── motorcycle/
  │   ├── images/
  │   └── labels/
  └── truck/
      ├── images/
      └── labels/
```

**Important**: Place your collected images in the `images/` subdirectory for each class.

## Quality Checklist
- [ ] Image is in focus
- [ ] Target object visible (even if small)
- [ ] Camera pointed straight down
- [ ] Metadata recorded (height, lighting, angle)
- [ ] Matches actual deployment scenario

## Building Color Consideration

⚠️ **IMPORTANT**: If your training building color differs from deployment building color:

**Risk**: Model may overfit to background color/contrast instead of object features

**Solutions**:
1. **Test First**: Collect 5-10 photos on deployment building, test baseline model
2. **If performance drops >20%**:
   - Collect equal photos from BOTH building colors (20 each per class)
   - OR increase color augmentation in training (already enabled)
   - OR train primarily on deployment building color

**Best Practice**: When possible, collect majority of photos on the **actual deployment building**

## Labeling Instructions

### How to Label Images

Use the interactive labeling tool to create YOLO format bounding boxes:

```bash
# Label motorcycle images (auto-detects class from directory name)
python3 label_images.py source_data/real_drone_photos/motorcycle

# Label other classes
python3 label_images.py source_data/real_drone_photos/bird
python3 label_images.py source_data/real_drone_photos/car
python3 label_images.py source_data/real_drone_photos/cat
python3 label_images.py source_data/real_drone_photos/dog
python3 label_images.py source_data/real_drone_photos/truck
```

### Labeling Controls
- **Click and drag** - Draw bounding box around object
- **n** - Next image (auto-saves current labels)
- **p** - Previous image (auto-saves current labels)
- **c** - Clear all labels for current image
- **u** - Undo last bounding box
- **s** - Save labels manually
- **q** - Quit (auto-saves)
- **0-5** - Change class:
  - 0=car, 1=motorcycle, 2=truck
  - 3=bird, 4=cat, 5=dog

The labeling script will:
1. Open the first motorcycle image
2. Auto-detect "motorcycle" as the class (class ID 1)
3. Allow you to draw bounding boxes around motorcycles
4. Save labels to `source_data/real_drone_photos/motorcycle/labels/`

## Next Steps
After collection and labeling:
1. Train model with combined dataset: `python train_with_real_data.py`
2. Validate predictions on real drone photos
3. Iterate based on failure cases
