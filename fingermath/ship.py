"""Notebook 06 helpers: variants of the browser pipeline and the data to train the shipped model.

Two knobs of the game's pipeline are tested here (both need a matching change in the JS):
- `line`: the pen width pad.renderForModel() draws with (game: 18 px on a 190 px digit);
- `resample`: how processing.js shrinks the 190 px digit to 20 px. 'nn' (nearest neighbour, the
  game today) keeps a ~2 px aliased, almost binary stroke; 'area' averages the pixels under each
  target pixel, which gives soft gray edges like the scanned MNIST digits.
"""
import json
from multiprocessing import Pool

import numpy as np
from PIL import Image

from . import datasets, synth
from .browser import render_strokes, js_round, otsu, shift_image

CACHE = datasets.CACHE


def preprocess(gray, resample='nn', center=True):
    """browser.preprocess_to_mnist() with a choice of resampling (processing.js `resample`)."""
    gray = np.asarray(gray, dtype=np.float32)
    th = otsu(gray)
    fg = (gray >= th) != (gray.mean() > 0.5)
    ys, xs = np.nonzero(fg)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        x0, y0, x1, y1 = 0, 0, gray.shape[1] - 1, gray.shape[0] - 1
    w, h = x1 - x0 + 1, y1 - y0 + 1
    scale = 20 / w if w > h else 20 / h
    new_w, new_h = max(1, int(js_round(w * scale))), max(1, int(js_round(h * scale)))
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    if resample == 'area':
        scaled = np.asarray(Image.fromarray(crop, 'F').resize((new_w, new_h), Image.BOX), np.float32)
    else:
        sx = np.minimum(w - 1, js_round(np.arange(new_w) / scale))
        sy = np.minimum(h - 1, js_round(np.arange(new_h) / scale))
        scaled = crop[np.ix_(sy, sx)]
    out = np.zeros((28, 28), np.float32)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    out[oy:oy + new_h, ox:ox + new_w] = scaled
    if center and out.sum() > 0:
        yy, xx = np.mgrid[0:28, 0:28]
        m = out.sum()
        dx, dy = int(js_round(14 - (out * xx).sum() / m)), int(js_round(14 - (out * yy).sum() / m))
        if dx or dy:
            out = shift_image(out, dx, dy)
    lo, hi = out.min(), out.max()
    return (out - lo) / max(1e-6, hi - lo)


def to28(strokes, line=18, resample='nn'):
    return preprocess(render_strokes(strokes, line=line), resample)


# ------------------------------------------------------------------ datasets (cached)

VARIANTS = {          # name -> (pen width, resampling)
    'nn18': (18, 'nn'),
    'area18': (18, 'area'),
    'area26': (26, 'area'),
}


def _skel_job(args):
    imgs, seed, pens = args
    rng = np.random.default_rng(seed)
    out = {v: [] for v in VARIANTS}
    out.update({v + '_penaug': [] for v in VARIANTS})
    for im in imgs:
        st = synth.skeleton_strokes(im, rng)
        for v, (line, rs) in VARIANTS.items():
            out[v].append(to28(st, line, rs))
            if pens:  # same skeleton, a random pen width around the game's one
                out[v + '_penaug'].append(to28(st, line * rng.uniform(0.6, 1.4), rs))
    return {k: np.stack(v) for k, v in out.items() if v}


def skeleton_sets(images, labels, name, seed, pens=False, chunk=500):
    """MNIST digits as pad strokes, through every pipeline variant at once."""
    path = CACHE / f'ship_{name}.npz'
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files if k != 'y'}, d['y']
    tasks = [(images[i:i + chunk], seed * 100003 + i, pens) for i in range(0, len(images), chunk)]
    with Pool() as pool:
        parts = pool.map(_skel_job, tasks)
    sets = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in parts[0]}
    np.savez_compressed(path, y=labels, **sets)
    return sets, labels


def _tmpl_job(args):
    digits, writer, seed = args
    rng = np.random.default_rng(seed)
    strokes = [synth.template_strokes(d, writer, rng) for d in digits]
    imgs = {v: np.stack([to28(s, line, rs) for s in strokes]) for v, (line, rs) in VARIANTS.items()}
    return imgs, strokes


def template_sets(n_writers, per_digit, name, seed):
    """Template writers (the held-out generator), every variant, plus the raw strokes
    (exported for the end-to-end browser test)."""
    path = CACHE / f'ship_{name}.npz'
    spath = CACHE / f'ship_{name}_strokes.json'
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files if k not in ('y', 'w')}, d['y'], d['w'], json.loads(spath.read_text())
    rng = np.random.default_rng(seed)
    tasks, ys, ws = [], [], []
    for wid in range(n_writers):
        writer = synth.make_writer(rng)
        digits = np.repeat(np.arange(10), per_digit)
        tasks.append((digits, writer, int(rng.integers(1 << 31))))
        ys.append(digits); ws.append(np.full(len(digits), wid))
    with Pool() as pool:
        parts = pool.map(_tmpl_job, tasks)
    sets = {v: np.concatenate([p[0][v] for p in parts]).astype(np.float32) for v in VARIANTS}
    strokes = [[np.round(s, 4).tolist() for s in st] for p in parts for st in p[1]]
    y, w = np.concatenate(ys), np.concatenate(ws)
    np.savez_compressed(path, y=y, w=w, **sets)
    spath.write_text(json.dumps(strokes))
    return sets, y, w, strokes
