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
| 0-9 | Pick the class (list below) |
| u | Undo the last box |
| c | Clear every box on this image |
| n / p | Next / previous image (saves) |
| q | Quit (saves) |

AVR 2026 classes:

| Key | Class | Key | Class |
|---|---|---|---|
| 0 | wheat_barrel | 5 | water_barn |
| 1 | water_barrel | 6 | bridge_1line |
| 2 | gasoline | 7 | bridge_2line |
| 3 | toxic_fluid | 8 | bridge_3line |
| 4 | wheat_barn | 9 | blackout |

## The rules

1. **Box every target in the frame, every time.** One image, all classes, one pass. An unboxed target tells the model "this is background", and in a test set it turns a correct detection into a false alarm.
2. **Box the artwork, not the holder.** For a barrel, box the printed sticker inside the can rim. For a barn, box the card you can see through the roof opening.
3. **Tight boxes.** Every edge touches the artwork, with no gap and nothing cut off.
4. **Cut off at the frame edge?** Box what you can see if you could name the class from it alone (about a third or more visible). If you can't tell what it is, leave it unboxed.
5. **Can't tell which class?** Skip the image (press `n` with no boxes) and write its name down. Don't guess.
6. **Nothing there?** An empty image is useful. Press `n` and move on.

## Check your work

After each session, look at your boxes, not the folder names. Render a contact sheet of the labeled images and scan for missed targets and wrong classes before anything gets trained.
