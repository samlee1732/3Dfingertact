# 3DFingerTact

Minimal offline image-to-3D reconstruction extracted from the 9DFingerTact
shape reconstruction pipeline.

## Run

```bash
git clone https://github.com/samlee1732/3Dfingertact.git 3DFingerTact
cd 3DFingerTact
conda activate 9dtact
python -m pip install -r requirements.txt
python shape_reconstruction/reconstruct.py
```

To use other rectified and cropped images, edit these variables at the top of
`shape_reconstruction/reconstruct.py`:

```python
REFERENCE_PATH = Path('/path/to/reference.png')
INPUT_PATH = Path('/path/to/input.png')
OUTPUT_PATH = Path('/path/to/output')
```

Each run creates a `results/YYMMDD-HHMMSS` directory containing numbered images
for every transformation stage, original and paper-style 3D point-cloud images,
stage metrics, and the exact intermediate arrays. The final `points` array in
`18-reconstruction.npz` has shape `N x 3`.
