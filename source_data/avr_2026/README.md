# AVR 2026 drone photos

142 captures, 640x480, taken 2026-09-27 over about ten minutes. Straight off
the drone camera, so the resolution and compression match what the model sees
in flight.

Contents: stickers seated in soup cans with a visible rim, barn cards recessed
behind a roof opening, black field surface with glare, gym floor with overhead
lights. Varied scale, rotation and motion blur.

## Why these matter

The shipped 2026 model was trained on **synthetic composites**: print artwork
pasted onto real background photographs through a random homography. The
backgrounds were real; the artwork was always flat.

Nothing in training had a can rim, a curved inset disc, specular glare, or a
card in shadow behind a roof. These are the first real photographs of mounted
stickers in the set.

## Where the current model stands on them

Scored with `avr2026n320.onnx` at its 0.45 floor, top detection per image:

```
at least one detection   124 / 142  (87%)
nothing above 0.45        18 / 142  (13%)

toxic_fluid   30  med 0.79      water_barrel  20  med 0.79
wheat_barrel  27  med 0.81      water_barn    20  med 0.87
wheat_barn    25  med 0.82      gasoline       2  med 0.77
```

These are unlabeled, so that is a firing rate, not accuracy. Some frames are
legitimately empty.

Two things it does show:

- **gasoline is the weak class on real mounts.** Two detections against 20-30
  for everything else, while gasoline cans appear plainly among the misses.
  One clean, centered, well-lit gasoline can scores **0.22**.
- **Several misses sit just under the line**, 0.40 to 0.44. An unambiguous
  water_barrel scores 0.44 against a 0.45 threshold.

## Next

1. Sort `raw/` into `real_drone_photos/<class>/images/`.
2. Label with `label_images.py`.
3. Retrain, as a separate change.

Sort from the artwork, not from model predictions. 120 frames were misfiled as
negatives during 2026 training and every one contained a wheat card; they would
have taught the model that wheat artwork is background across a quarter of the
dataset.

The misses and the 0.40-0.44 near-misses are worth labeling first. Labeling all
142 is hours; labeling the ~30 that the model gets wrong is an afternoon and
targets the actual gap.
