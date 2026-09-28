# AVR 2026 drone photos

142 captures, 640x480, taken 2026-09-27 over about ten minutes. Straight off
the drone camera, so the resolution and compression match what the model sees
in flight.

Contents: stickers seated in soup cans with a visible rim, barn cards recessed
behind a roof opening, black field surface with glare, gym floor with overhead
lights. Varied scale, rotation and motion blur. Barrels and barns only — no
bridge or blackout cards in this set.

## Why these matter

The `avr-yolo` generator is not naive. Its 2026 dataset is about 60% real-backed
already: 24% of frames are pure real photographs used as negatives, and another
26% paste the artwork onto real background photos through a homography with
occlusion, truncation and varied blend seams. The remaining 40% are renders of
the CAD field through the camera's measured intrinsics, and those renders do
model the physical mounts — a 3 in can mouth with the image 4.5 in down a tin
body, the barn shafts, the bridge cups.

What the set has never contained is **a real photograph carrying a positive
label**. Every labeled box in training is rendered or pasted. Real photographs
appear only as backgrounds and as "nothing here."

That is the leg these images add, and it is the same leg the 2025 pipeline had:
augmented artwork plus real drone photos, blended.

## Where the current model stands on them

Scored with `avr2026n320_v6/best.onnx` at the deployed 0.45 floor, per-class NMS,
counting every surviving box:

```
images 142   with at least one detection 124 (87%)

class           boxes  images  best conf  in 0.30-0.45
wheat_barrel       54      53       0.88             4
water_barrel       60      52       0.86             8
toxic_fluid        59      58       0.88             5
wheat_barn         29      29       0.91             0
water_barn         26      25       0.92             2
gasoline           21      21       0.82             8
bridge_*            0       0       0.32             1
blackout            0       0       0.06             0
```

Unlabeled, so this is a firing rate, not accuracy. Bridge and blackout are zero
because those cards are not in these photos.

The one asymmetry worth chasing: **gasoline fires on 21 images against 52-58 for
the other three barrel classes, with 8 more sitting in 0.30-0.45.** That is not
the same as the 119/120 `avr-yolo` measured on `testdata/real_gasoline`, which
was a printed card on an Arducam at 1280x720. These are cans, on the aircraft's
own 640x480 lens.

69 of the 142 frames hold two or more objects, which is why the images are one
flat set rather than a directory per class: every object in a frame gets a box,
or the unmarked ones train as background.

## Next

```bash
pip install -r requirements.txt
python3 label_images.py source_data/avr_2026/captures --class gasoline
```

Labels land in `captures/labels/`, one `.txt` per image. Number keys switch
class mid-image, `n` advances and saves, `q` quits and saves.

Then retrain, as a separate change.

Work from the artwork, not from what a model predicts. `avr-yolo` filed 120
frames as negatives during 2026 training and every one contained a wheat card;
they would have taught the model that wheat artwork is background across a
quarter of the dataset. Render a labeled contact sheet before anything trains
on it.

Hold a slice back, split by capture run rather than at random. These frames are
consecutive, so neighbors are near-duplicates and a random split leaks. If all
142 train, nothing real is left to measure against.
