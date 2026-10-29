"""Notebook 09: stroke-level augmentation of real handwriting, drawn through the shipped pipeline.

A real sample is a list of strokes; each variant applies one random affine transform to all
of them (rotation, horizontal shear for slant, separate x/y scale) and draws the result with a
random pen width, then `ship.to28(..., 'area')` turns it into the 28x28 model input.
"""
import numpy as np

from . import ship

ROT_DEG = 10        # rotation, +- degrees
SHEAR = 0.35        # horizontal shear (slant), +-
SCALE = (0.8, 1.2)  # x and y scale, each drawn separately
PEN = (12, 26)      # pen width inside the 190 px digit box (the game draws 18)


def transform(strokes, rng):
    a = np.deg2rad(rng.uniform(-ROT_DEG, ROT_DEG))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shear = np.array([[1.0, rng.uniform(-SHEAR, SHEAR)], [0.0, 1.0]])   # x += s * y
    scale = np.diag(rng.uniform(*SCALE, size=2))
    m = rot @ shear @ scale
    centre = np.concatenate(strokes).mean(0)
    return [(np.asarray(s, float) - centre) @ m.T for s in strokes]


def render_variants(strokes, seed, n=4):
    """The sample as written (game pen) followed by `n` random variants: (n + 1, 28, 28) float32."""
    rng = np.random.default_rng(seed)
    out = [ship.to28(strokes, 18, 'area')]
    for _ in range(n):
        out.append(ship.to28(transform(strokes, rng), rng.uniform(*PEN), 'area'))
    return np.stack(out).astype(np.float32)
