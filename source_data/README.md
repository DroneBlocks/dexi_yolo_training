# source_data layout

One directory per AVR season from 2026 onward.

```
source_data/
  original_images/          AVR 2025. Hand-labeled stock images, 6 COCO classes.
  real_drone_photos/        AVR 2025. Photos off the drone, same 6 classes.
  avr_2026/
    raw/                    Captures straight off the drone, not yet sorted.
    real_drone_photos/      Sorted and labeled, one directory per class.
      <class>/images/
      <class>/labels/
```

2025 sits at the top level because it predates this convention and because
`train_with_real_data.py` hardcodes `source_data/real_drone_photos`. Moving it
belongs with the change that makes that path configurable, not here.

A new season adds `avr_<year>/` in the same shape.

## Classes

| season | classes |
|---|---|
| 2025 | car, motorcycle, truck, bird, cat, dog |
| 2026 | wheat_barrel, water_barrel, gasoline, toxic_fluid, wheat_barn, water_barn, bridge_1line, bridge_2line, bridge_3line, blackout |

Order matters when training: the model emits indices and the class list is the
only thing that names them. `dexi_yolo/models/models.yaml` holds the order each
shipped model was trained in.

## Labeling

`label_images.py` takes a directory and reads the class from its name, so it
works anywhere:

```bash
python3 label_images.py source_data/avr_2026/real_drone_photos/gasoline
```

Sort `raw/` into the class directories first. Sort by looking at the artwork,
not by what a model predicts: a model-sorted set teaches the next model what
the last one already believed.
