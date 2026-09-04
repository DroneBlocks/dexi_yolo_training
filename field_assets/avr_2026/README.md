# AVR 2026 field artwork, "Runway to Roots"

Print-ready artwork for the 2026 AVR field, plus one clean image per detection
class for YOLO training.

Anyone building a practice field at their school can print from `print/` and get
the same stickers the competition field uses. Everything here is 300 dpi or
vector, so it prints at full size without softening.

## What is in here

```
print/       The five source PDFs exactly as supplied. Print from these.
classes/     One clean image per detection class, 300 dpi, transparent PNG.
reference/   The tower and helipad sheets rendered as flat PNGs, for looking at.
```

`classes/` is what you train on. `print/` is what you send to a printer.

## The detection classes

Each sticker tells the drone what to do. The drone reads the sticker and lights
its LED, or deliberately does nothing.

| Class | Where it appears | Printed size | DEXI-5 LED |
|---|---|---|---|
| `wheat` | Barrel, and block inside a barn roof | 2 5/8" circle, 4" square | Green |
| `water` | Barrel, and block inside a barn roof | 2 5/8" circle, 4" square | Blue |
| `gasoline` | Barrel and train car | 2 5/8" circle | Red |
| `toxic` | Barrel and train car | 2 5/8" circle | Yellow |
| `blackout` | Block inside a barn roof, and train car | 4" square, 2 5/8" circle | None |
| `hay_bale` | Block inside a barn roof | 4" square | None |
| `bridge_1` | Bridge cup | 3" circle | Red |
| `bridge_2` | Bridge cup | 3" circle | Green |
| `bridge_3` | Bridge cup | 3" circle | Blue |
| `bridge_blank` | Bridge cup, bottom of each of the 4 cups | 3" circle | None, do not seed |

The classes with no LED are the ones that matter most. A model that lights up on
a blackout sticker costs points, so those need as many training photos as the
positive classes, not fewer.

## Two things to know before you train

**1. `hay_bale` has no artwork yet.** It is in the sticker spec at 4" x 4" and it
is one of the no-LED classes, but it was not in the supplied set. There is no
`classes/hay_bale/` folder because there is nothing to put in it. That class
cannot be trained until the artwork arrives.

**2. `blackout` and `bridge_blank` are the same graphic.** Both are a plain black
circle, one at 2 5/8" and one at 3". A model cannot tell them apart from the
image alone, only from where it is looking. If you train them as two classes you
will get them confused for each other. Consider training one `black_circle` class
and deciding what it means from context.

## Sizes, measured

Measured off the rendered artwork rather than read off the spec:

| File | Ink size | Page size |
|---|---|---|
| `wheat/barrel_2.625in.png` | 2.627" (66.7 mm) | 3" |
| `water/barrel_2.625in.png` | 2.627" (66.7 mm) | 3" |
| `gasoline/barrel_2.625in.png` | 2.627" (66.7 mm) | 3" |
| `toxic/barrel_2.625in.png` | 2.627" (66.7 mm) | 3" |
| `blackout/barrel_2.625in.png` | 2.627" (66.7 mm) | 3" |
| `wheat/block_4in.png` | 3.603" of ink on a 4" sticker | 4" |
| `water/block_4in.png` | 4.000" (101.6 mm) | 4" |
| `blackout/block_4in.png` | 4.000" (101.6 mm) | 4" |
| `bridge_*/cup_3in.png` | 3.000" (76.2 mm) | 4" |

The barrel stickers land on 2 5/8" exactly. The bridge cups are 3" of artwork
centred on a 4" page, so trim to the circle rather than to the page edge.

## Printing

Print the PDFs in `print/` at 100%, with no scaling and no "fit to page". Scaling
is the usual way a field ends up with stickers that are the wrong size, and
sticker size is what sets the altitude the drone can read them from.

- `Barrels.pdf` 5 pages: wheat, water, gasoline, toxic, blackout
- `Blocks.pdf` 3 pages: wheat, water, blackout
- `Bridge_Numerals.pdf` 4 pages: #1, #2, plain black, #3
- `DEXI 5 Tower Bottom.pdf` and `DEXI 5 Tower Top + Helipads.pdf` are large
  format, 52" x 55" and 50" x 48". These are for a print shop, not an office
  printer.

**AprilTags are not in this folder.** Print those from `DroneBlocks/dexi_apriltag`
using `tags/apriltag_00000.png` through `apriltag_00006.png`. Scale so the black
square measures 6", which puts the sheet at 7.5", then trim the white to 7". Use
nearest-neighbour scaling so the cell edges stay hard; smooth scaling blurs the
corners the detector relies on.

## Using these for training

The images in `classes/` are clean, flat, straight-on renders. They are the
starting point, not the whole dataset. The known failure on this field is a
sticker seen from altitude, at an angle, recessed inside a barn roof, and a model
trained only on flat renders does not survive that.

The workflow in the repository root handles this: `augment_dataset.py` generates
perspective and lighting variations from a clean source image, and
`train_with_real_data.py` blends those with real photos taken from the drone.
Read the root `README.md` for the full sequence.

Two things about this year's classes in particular:

- Last year's six classes were all COCO classes, so `yolov8n.pt` already
  half-knew them. `wheat`, `hay_bale`, `toxic` and `blackout` are not in COCO.
  Real photos from the drone matter more this year, and 50 epochs may not be
  enough.
- The 2 5/8" barrel sticker seen from altitude through a 320 x 320 network input
  is the hard case. Find the altitude where it stops being readable before
  tuning anything else. If that altitude is low, it is a field design input, not
  a model problem.
