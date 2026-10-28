"""Notebooks 12+: the training data and budget of notebook 09's chosen recipe (config C), in one place.

Config C = MNIST (augmented) 40% + synthetic game strokes 40% + real Pendigits train strokes 20%
(each sample drawn as written plus 4 random stroke-level variants, see augstrokes.py), every batch
mixed in those shares; 16 epochs x 470 steps x 128, Adam with cosine decay from 2e-3, label smoothing 0.1.
"""
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import keras

import augstrokes, datasets, models, ship

REAL = Path('cache/real')
EPOCHS, STEPS, BATCH, SMOOTHING, LR = 16, 470, 128, 0.1, 2e-3
SEEDS = [42, 43]


def real_meta():
    d = np.load(REAL / 'real_digits.npz')
    meta = pd.DataFrame({k: d[k] for k in ['source', 'device', 'writer', 'split']})
    meta['label'] = d['y']
    return meta, d['x'].astype(np.float32) / 255


def eval_splits(meta):
    """Row numbers of the real validation and test sets (notebook 07 splits)."""
    part = lambda src, split: np.flatnonzero((meta.source == src) & (meta.split == split))
    return {'val: pen': part('pendigits', 'val'), 'val: finger': part('corr', 'dev'),
            'test: pen': part('pendigits', 'test'), 'test: finger': part('corr', 'test')}


def strokes():
    with open(REAL / 'real_strokes.pkl', 'rb') as f:
        return [s['strokes'] for s in pickle.load(f)]


def pen_train_renders(meta, all_strokes):
    """Pendigits train, 5 images per sample (as written + 4 variants), exactly as notebook 09; cached."""
    path = REAL / 'pen_train_renders.npz'
    if path.exists():
        z = np.load(path)
        return z['x'], z['y']
    k = np.flatnonzero((meta.source == 'pendigits') & (meta.split == 'train'))
    with Pool() as pool:
        r = pool.starmap(augstrokes.render_variants, [(all_strokes[i], 1000 + j) for j, i in enumerate(k)], chunksize=100)
    x, y = np.concatenate(r).astype(np.float32), np.repeat(meta.label.to_numpy()[k], 5)
    np.savez_compressed(path, x=x, y=y)
    return x, y


def sources_c(meta, all_strokes):
    x_mn, y_mn, _, _ = datasets.load_mnist()
    x_mn = x_mn.astype(np.float32) / 255
    skel, y_skel = ship.skeleton_sets(None, None, 'skel_train', seed=1, pens=True)   # cached by notebook 06
    x_pen, y_pen = pen_train_renders(meta, all_strokes)
    return [(x_mn, y_mn, models.augment_mnist, 0.4), (skel['area18'], y_skel, models.augment_light, 0.4),
            (x_pen, y_pen, models.augment_light, 0.2)]


def compile_and_fit(model, ds, seed, verbose=0):
    """Config C's optimiser and budget; `ds` yields (x, one-hot y)."""
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(LR, STEPS * EPOCHS)),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=SMOOTHING))
    model.fit(ds, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=verbose)
    return model
