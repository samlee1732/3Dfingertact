# Figure 3 9DTact logo images

`reference.png` and `input.png` were extracted from the official 9DTact
pipeline figure and resized to the 460 x 345 input size used by this project.

Source:
https://github.com/linchangyi1/9DTact/blob/main/source/pipelie.png

Crop coordinates in the 11575 x 4015 source image:

- input: `(1999, 392, 3180, 1279)`
- reference: `(1999, 2392, 3180, 3279)`

The official project does not publish these two frames as standalone raw sensor
images. These files therefore contain figure-export and resize artifacts. They
can be used for a qualitative reconstruction of the 9DTact logo, but not for
quantitative depth evaluation.
