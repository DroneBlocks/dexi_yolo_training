# Labeling Guide

How to draw boxes on real drone photos so they can train or test a model.

## Start the labeler

```bash
python3 label_images.py source_data/avr_2026/captures
python3 label_images.py source_data/avr_2026/captures --start 28   # resume at image 28
```

| Key | Action |
|---|---|
| drag | Draw a box |
| 0-5 | Pick the class (list below) |
| u | Undo the last box |
| c | Clear every box on this image |
| n / p | Next / previous image (saves) |
| q | Quit (saves) |
| h | Hide or show the info panel |
| f | Flag an unsure frame and set it aside |

AVR 2026 classes:

| Key | Class | Key | Class |
|---|---|---|---|
| 0 | wheat_barrel | 3 | toxic_fluid |
| 1 | water_barrel | 4 | wheat_barn |
| 2 | gasoline | 5 | water_barn |

Bridge cups and blackout circles are not labeled. Counting bridge bars is done by a separate OpenCV node, and a blackout circle is an empty slot with nothing to report. Leave them unboxed: to the detector they are background.

## The rules

1. **Box every target in the frame, every time.** One image, all classes, one pass. An unboxed target tells the model "this is background", and in a test set it turns a correct detection into a false alarm.
2. **Box the artwork, nothing else.** The printed artwork is what tells one class from another, so it is the object. For a barrel, box the white sticker inside the can: no rim, no can, no shadow. For a barn, box the card you can see through the roof opening. Training data and test set both follow this rule, so every model is graded against the same shape.
3. **Tight boxes.** Every edge touches the artwork, with no gap and nothing cut off.
4. **Cut off at the frame edge?** Box what you can see if you could name the class from it alone (about a third or more visible). If you can't tell what it is, leave it unboxed.
5. **Can't tell which class?** Press `f` to flag the frame. Don't guess, and don't press `n` with no boxes: that saves the frame as empty, which says there is nothing in it.
6. **Nothing there?** An empty image is useful. Press `n` and move on.

## Check your work

After each session, look at your boxes, not the folder names. Render a contact sheet of the labeled images and scan for missed targets and wrong classes before anything gets trained.
