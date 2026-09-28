# source_data layout

One directory per AVR season from 2026 onward.

```
source_data/
  original_images/          AVR 2025. Hand-labeled stock images, 6 COCO classes.
  real_drone_photos/        AVR 2025. Photos off the drone, same 6 classes.
  avr_2026/
    classes.txt             Class names in training order.
    captures/
      images/               Photos off the drone.
      labels/               One .txt per image, every object in the frame.
```

2025 sits at the top level because it predates this convention and because
`train_with_real_data.py` hardcodes `source_data/real_drone_photos`. Moving it
belongs with the change that makes that path configurable, not here.

A new season adds `avr_<year>/` in the same shape.

## Why 2026 is flat and 2025 is per-class

2025 photos hold one object each, so a directory per class was also the label.
2026 frames hold several: 69 of the first 142 have two or more objects. A
directory per class would mean labeling one barrel per frame and leaving the
others unmarked, which trains the model that they are background. So the images
stay in one set and every object in a frame gets a box.

## Classes

| season | classes |
|---|---|
| 2025 | car, motorcycle, truck, bird, cat, dog |
| 2026 | wheat_barrel, water_barrel, gasoline, toxic_fluid, wheat_barn, water_barn, bridge_1line, bridge_2line, bridge_3line, blackout |

Order matters when training: the model emits indices and the class list is the
only thing that names them. `dexi_yolo/models/models.yaml` holds the order each
shipped model was trained in, and `classes.txt` mirrors it.

## Labeling

`label_images.py` reads `classes.txt` from the set being labeled, or the
nearest one above it, and falls back to the 2025 list when there is none.

```bash
python3 label_images.py source_data/avr_2026/captures --class gasoline
```

`--class` only sets which class the next box gets; the number keys change it
mid-image. Box every object you can see, not just the one you started on.

Work from the artwork, not from what a model predicts. A model-sorted set
teaches the next model what the last one already believed.
