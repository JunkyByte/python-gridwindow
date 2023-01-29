# python-gridwindow
A mini python library that provides grid visualization of multiple images/videos in a single opencv window.  
Suppose you want to see multiple videos at same time in a single opencv window. This library will take care of placement and scaling of the videos for you, even when a disposition is not trivial.  
The aspect ratio will be preserved but the videos might be scaled down to fit inside the `(max_width, max_height)` specified.

## Installation
```
git clone https://github.com/JunkyByte/python-gridwindow.git
cd python-gridwindow/
pip install .
```

## Usage
A single class `MagicGrid` is exposed.
```python
from gridwindow import MagicGrid
window = MagicGrid(max_width=1280, max_height=720)

while True:
    # images = ...
    if window.update(images) & 0xFF == ord('q'):
        break
```

Check `gridwindow/grid_window.py:MagicGrid` and `demo.py` for more details.

## How?
In the end this project needs to solve a rectangle packing problem which is NP hard.
To find a solution we use [rpack](https://github.com/Penlect/rectangle-packer).


## Why?
I wrote a grid based window for another project with multiple cameras operating at the same time.
I wondered how it could be adapted to support any number of videos and resolutions. This was fun.
