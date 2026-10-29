"""Datasets shared by the research notebooks (cached in ./cache, which git ignores)."""
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import synth
from .browser import strokes_to_28

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / 'cache'
SAMPLES = ROOT / 'web' / 'collect' / 'samples'


def load_mnist():
    """MNIST from data/MNIST.zip -> (x_train, y_train, x_test, y_test); x is uint8 (N, 28, 28)."""
    path = CACHE / 'mnist.npz'
    if path.exists():
        d = np.load(path)
        return d['x_train'], d['y_train'], d['x_test'], d['y_test']
    out = {}
    with zipfile.ZipFile(ROOT / 'data' / 'MNIST.zip') as z:
        for name in ['xtrain', 'ytrain', 'xtest', 'ytest']:
            with z.open(f'digit_{name}.csv') as f:
                out[name] = pd.read_csv(f, header=None).values
    x_train = out['xtrain'].reshape(-1, 28, 28).astype(np.uint8)
    x_test = out['xtest'].reshape(-1, 28, 28).astype(np.uint8)
    y_train, y_test = out['ytrain'].ravel(), out['ytest'].ravel()
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def _parallel(job, tasks, processes=None):
    with Pool(processes) as pool:
        return np.concatenate(pool.map(job, tasks))


def pad_skeleton_set(images, labels, name, seed=0, jitter=True, chunk=500):
    """MNIST digits redrawn with the pad's pen and passed through the browser pipeline."""
    path = CACHE / f'{name}.npz'
    if path.exists():
        d = np.load(path)
        return d['x'], d['y']
    tasks = [(images[i:i + chunk], seed * 100003 + i, jitter) for i in range(0, len(images), chunk)]
    x = _parallel(synth._skeleton_job, tasks).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, x=x, y=labels)
    return x, labels


def _pipeline_job(imgs):
    """Upscale MNIST to the pad's 280x280 and run the browser preprocessing on it."""
    from PIL import Image
    from .browser import preprocess_to_mnist
    return np.stack([preprocess_to_mnist(np.asarray(Image.fromarray(im).resize((280, 280), Image.BILINEAR),
                                                    np.float32) / 255) for im in imgs])


def mnist_through_pipeline(images, name, chunk=500):
    """MNIST images passed through the browser preprocessing (no change of handwriting)."""
    path = CACHE / f'{name}.npz'
    if path.exists():
        return np.load(path)['x']
    x = _parallel(_pipeline_job, [images[i:i + chunk] for i in range(0, len(images), chunk)])
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32))
    return x


def template_set(n_writers, per_digit, name, seed=0, noise=1.0, switch=0.0):
    """Template-stroke digits: n_writers writers x 10 digits x per_digit samples.

    Returns x (N, 28, 28), y (N,), writer id (N,).
    """
    path = CACHE / f'{name}.npz'
    if path.exists():
        d = np.load(path)
        return d['x'], d['y'], d['w']
    rng = np.random.default_rng(seed)
    tasks, ys, ws = [], [], []
    for wid in range(n_writers):
        writer = synth.make_writer(rng, noise, switch)
        digits = np.repeat(np.arange(10), per_digit)
        tasks.append((digits, writer, int(rng.integers(1 << 31))))
        ys.append(digits)
        ws.append(np.full(len(digits), wid))
    x = _parallel(synth._template_job, tasks).astype(np.float32)
    y, w = np.concatenate(ys), np.concatenate(ws)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, x=x, y=y, w=w)
    return x, y, w


def load_real_samples(single_digit=True):
    """Samples saved by collect/index.html (every *.json in collect/samples/).

    Returns a list of dicts (label, writer, device, strokes, pad, img28, file); the
    28x28 image is recomputed in Python when the page did not store one.
    """
    out = []
    for f in sorted(SAMPLES.glob('*.json')):
        for s in json.loads(f.read_text(encoding='utf-8')):
            if single_digit != (len(s['label']) == 1):
                continue
            s['strokes'] = [np.asarray(st, dtype=np.float64) for st in s['strokes']]
            s['file'] = f.name
            if single_digit:
                s['img28_py'] = strokes_to_28(s['strokes'], s['pad'])
                s['img28'] = (np.asarray(s['img28'], np.float32).reshape(28, 28)
                              if s.get('img28') else s['img28_py'])
            out.append(s)
    return out
