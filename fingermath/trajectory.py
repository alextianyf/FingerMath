"""Notebook 11: the pen path as model input (online recognition), next to the 28x28 image.

`encode` turns a digit's strokes (in writing order, x right, y down) into a fixed-length
sequence: N points spaced evenly along the ink, shared out between the strokes by their
length, with 5 features per point:

    x, y        position, centred on the bounding box and scaled by its longer side (aspect kept)
    dx, dy      step to the next point of the same stroke (0 at a stroke's last point), x N / 4
    pen_down    1 at the first point of every stroke after the first (the pen was lifted before it)

Timing is left out on purpose: pen, finger and mouse move at very different speeds.
"""
import numpy as np
import pandas as pd

import augstrokes

N = 64
FEATURES = 5


def _resample(stroke, n):
    s = np.asarray(stroke, float).reshape(-1, 2)
    if len(s) == 1 or n == 1:
        return np.repeat(s[:1], n, 0)
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)]
    if t[-1] < 1e-9:
        return np.repeat(s[:1], n, 0)
    u = np.linspace(0, t[-1], n)
    return np.stack([np.interp(u, t, s[:, 0]), np.interp(u, t, s[:, 1])], 1)


def _share(lengths, n):
    """Points per stroke: proportional to length, at least 2 (1 for a dot), summing to n."""
    k = len(lengths)
    if k >= n:                                    # absurdly many strokes: keep the longest
        keep = np.argsort(-lengths)[:n // 2]
        out = np.zeros(k, int); out[keep] = 2
        out[keep[0]] += n - out.sum()
        return out
    base = np.where(lengths > 0, 2, 1)
    rest = n - base.sum()
    w = lengths / max(lengths.sum(), 1e-9)
    extra = np.floor(w * rest).astype(int)
    order = np.argsort(-(w * rest - extra))       # largest remainders get the leftover points
    extra[order[:rest - extra.sum()]] += 1
    return base + extra


def encode(strokes, n=N):
    strokes = [np.asarray(s, float).reshape(-1, 2) for s in strokes if len(s)]
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    centre, scale = (lo + hi) / 2, max((hi - lo).max(), 1e-6)
    lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes])
    out = []
    for j, (s, k) in enumerate(zip(strokes, _share(lengths, n))):
        if k == 0:
            continue
        r = (_resample(s, k) - centre) / scale
        step = np.vstack([np.diff(r, axis=0), [0, 0]]) * (n / 4)
        down = np.zeros((k, 1)); down[0] = 1.0 if out else 0.0
        out.append(np.hstack([r, step, down]))
    return np.concatenate(out)[:n].astype(np.float32)


def encode_variants(strokes, seed, n_aug):
    """The sample as written followed by `n_aug` random affine variants (augstrokes.transform)."""
    rng = np.random.default_rng(seed)
    return np.stack([encode(strokes)] + [encode(augstrokes.transform(strokes, rng)) for _ in range(n_aug)])


def corr_strokes_in_order(raw_dir):
    """Corr strokes per valid glyph, in the same glyph order as notebook 07, but with the
    strokes in writing order: stroke.Z_FOK_GLYPH is Core Data's key for the ordered
    glyph -> strokes relation (notebook 07 sorted by ZINDEX, a table-wide id)."""
    glyph = pd.read_csv(raw_dir / 'glyph.csv')
    stroke = pd.read_csv(raw_dir / 'stroke.csv')
    touch = pd.read_csv(raw_dir / 'touch.csv')
    glyph = glyph[glyph.ZVALID == 1]
    touch = touch.sort_values(['ZSTROKE', 'ZTIMESTAMP'])
    stroke = stroke.sort_values(['ZGLYPH', 'Z_FOK_GLYPH'])
    pts_of = {k: g[['ZX', 'ZY']].to_numpy(float) for k, g in touch.groupby('ZSTROKE')}
    strokes_of = {k: list(g['Z_PK']) for k, g in stroke.groupby('ZGLYPH')}
    out = []
    for r in glyph.itertuples():
        ids = [i for i in strokes_of.get(r.Z_PK, []) if i in pts_of]
        if ids:
            out.append([pts_of[i] for i in ids])
    return out
