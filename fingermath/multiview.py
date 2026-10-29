"""Notebook 12b: every view of one handwritten digit, made from the same strokes.

- image:   the game's 28x28 model input (ship.to28, pen 18, 'area'): ink only;
- traj:    the pen path, 64 points x 5 features (trajectory.encode);
- ordermap: 3 extra 28x28 channels aligned pixel for pixel with the image, drawn with the same
           pen and put through the same crop / scale / centre-of-mass shift:
             dir_x, dir_y   the direction the pen was moving through that pixel (unit vector),
             order          how far along the whole path (all strokes, in writing order) it was, 0..1.
           Each is multiplied by the ink coverage (the shrink averages it with the black background),
           so where there is no ink all three are 0. Later strokes paint over earlier ones.

The idea of the order map: a plain image CNN that also sees *where the pen went and in which order*,
without a second network. Direction-decomposed feature maps are a classic of online Chinese
character recognition [31, 32]; adding the writing order as a channel and dropping the extra
channels during training so the same network still reads plain images (ModDrop [34]) is our own
combination for this game.
"""
import numpy as np
from PIL import Image, ImageDraw

from . import augstrokes, ship, trajectory
from .browser import MODEL_BOX, MODEL_LINE, MODEL_SIZE, js_round, otsu, render_strokes, shift_image


def _params(gray):
    """ship.preprocess(resample='area') split into 'measure' and 'apply'; this half measures."""
    th = otsu(gray)
    fg = (gray >= th) != (gray.mean() > 0.5)
    ys, xs = np.nonzero(fg)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        x0, y0, x1, y1 = 0, 0, gray.shape[1] - 1, gray.shape[0] - 1
    w, h = x1 - x0 + 1, y1 - y0 + 1
    scale = 20 / w if w > h else 20 / h
    nw, nh = max(1, int(js_round(w * scale))), max(1, int(js_round(h * scale)))
    return x0, y0, x1, y1, nw, nh


def _apply(img, p):
    x0, y0, x1, y1, nw, nh = p
    crop = np.ascontiguousarray(img[y0:y1 + 1, x0:x1 + 1], np.float32)
    out = np.zeros((28, 28), np.float32)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(crop, 'F').resize((nw, nh), Image.BOX), np.float32)
    return out


def _com_shift(ink):
    yy, xx = np.mgrid[0:28, 0:28]
    m = ink.sum()
    if m <= 0:
        return 0, 0
    return int(js_round(14 - (ink * xx).sum() / m)), int(js_round(14 - (ink * yy).sum() / m))


def _paint(strokes, values, line=MODEL_LINE, box=MODEL_BOX, size=MODEL_SIZE):
    """Draw every segment with its own value (one float image per channel), pad.js geometry, no AA."""
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    scale = box / max((hi - lo).max(), 1e-6)
    c = (lo + hi) / 2
    imgs = [Image.new('F', (size, size), 0.0) for _ in range(values[0].shape[1])]
    draws = [ImageDraw.Draw(im) for im in imgs]
    r = line / 2
    for s, v in zip(strokes, values):
        p = (s - c) * scale + size / 2
        for i in range(len(p)):
            a, b = p[i], p[min(i + 1, len(p) - 1)]
            for ch, d in enumerate(draws):
                d.ellipse((a[0] - r, a[1] - r, a[0] + r, a[1] + r), fill=float(v[i, ch]))
                if i + 1 < len(p):
                    d.line([tuple(a), tuple(b)], fill=float(v[i, ch]), width=int(round(line)))
    return [np.asarray(im, np.float32) for im in imgs]


def _point_values(strokes):
    """Per point: direction of the segment leaving it (the last point keeps the one before) and path position."""
    lens = [np.linalg.norm(np.diff(s, axis=0), axis=1) for s in strokes]
    total = max(sum(l.sum() for l in lens), 1e-9)
    done, out = 0.0, []
    for s, l in zip(strokes, lens):
        d = np.diff(s, axis=0)
        d = np.vstack([d, d[-1:]]) if len(d) else np.zeros((1, 2))
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(n > 1e-9, d / np.maximum(n, 1e-9), 0.0)
        t = done + np.r_[0, np.cumsum(l)]
        out.append(np.hstack([d, (t / total)[:, None]]))
        done += l.sum()
    return out


def views(strokes, line=18):
    """(image 28x28, ordermap 28x28x3, traj 64x5) for strokes in writing order."""
    strokes = [np.asarray(s, float).reshape(-1, 2) for s in strokes if len(s)]
    gray = render_strokes(strokes, line=line)
    image = ship.to28(strokes, line, 'area')
    p = _params(gray)
    ink = _apply(gray, p)
    dx, dy = _com_shift(ink)
    chans = [shift_image(_apply(ch, p), dx, dy) for ch in _paint(strokes, _point_values(strokes), line=line)]
    return image, np.stack(chans, -1), trajectory.encode(strokes)


def pen_views(strokes, seed, n=4):
    """notebook 09's 5 renders of a Pendigits train sample (augstrokes.render_variants, same random
    draws, so the images are identical), with the matching order maps and trajectories."""
    rng = np.random.default_rng(seed)
    out = [views(strokes, 18)]
    for _ in range(n):
        st = augstrokes.transform(strokes, rng)
        out.append(views(st, rng.uniform(*augstrokes.PEN)))
    return tuple(np.stack(v) for v in zip(*out))


# ------------------------------------------------------------------ training data for notebook 12b

def joint_ds(image_sources, paired_sources, keys, batch=128, seed=0, moddrop=0.3):
    """Batches of {'img', 'omap', 'traj', 'avail'} (only `keys`), y one-hot.

    image_sources:  (x (n,28,28), y, augment_fn, fraction): no pen path, so omap / traj are zeros, avail 0.
    paired_sources: (img, omap (n,28,28,3), traj, y, fraction): the image and its order map get the
                    same random affine transform; with probability `moddrop` a sample's path is hidden
                    (zeros, avail 0) so the model keeps working on the image alone [34].
    """
    import tensorflow as tf
    from . import models

    def finish(img, om, tr, av, y):
        plane = tf.ones_like(img) * av[:, None, None, :]
        d = {'img': img, 'omap': tf.concat([om * av[:, None, None, :], plane], -1), 'traj': tr * av[:, None, :], 'avail': av}
        return {k: d[k] for k in keys}, tf.one_hot(y, 10)

    parts = []
    for i, (x, y, aug, frac) in enumerate(image_sources):
        ds = tf.data.Dataset.from_tensor_slices((x[..., None].astype(np.float32), y.astype(np.int32)))
        ds = ds.shuffle(len(x), seed=seed + i).repeat().batch(max(1, round(batch * frac)))
        def img_only(a, b, f=aug):
            a = f(a); n = tf.shape(a)[0]
            return a, tf.zeros([n, 28, 28, 3]), tf.zeros([n, 64, 5]), tf.zeros([n, 1]), b
        parts.append(ds.map(img_only, num_parallel_calls=tf.data.AUTOTUNE))
    for j, (img, om, tr, y, frac) in enumerate(paired_sources):
        ds = tf.data.Dataset.from_tensor_slices((img[..., None].astype(np.float16), om.astype(np.float16),
                                                 tr.astype(np.float32), y.astype(np.int32)))
        ds = ds.shuffle(len(img), seed=seed + 100 + j).repeat().batch(max(1, round(batch * frac)))
        def paired(a, o, t, b):
            both = models._affine(tf.concat([tf.cast(a, tf.float32), tf.cast(o, tf.float32)], -1), training=True)
            keep = tf.cast(tf.random.uniform([tf.shape(a)[0], 1]) >= moddrop, tf.float32)
            return tf.clip_by_value(both[..., :1], 0, 1), both[..., 1:], t, keep, b
        parts.append(ds.map(paired, num_parallel_calls=tf.data.AUTOTUNE))
    ds = tf.data.Dataset.zip(tuple(parts)).map(lambda *p: finish(*[tf.concat([q[k] for q in p], 0) for k in range(5)]))
    return ds.prefetch(tf.data.AUTOTUNE)


def eval_inputs(img, omap, traj, keys, avail=1.0):
    """Model inputs for real samples, with the pen path (avail=1) or as plain images (avail=0)."""
    n = len(img)
    av = np.full((n, 1), avail, np.float32)
    om4 = np.concatenate([omap * avail, np.broadcast_to(av[:, None, None, :], (n, 28, 28, 1))], -1).astype(np.float32)
    d = {'img': img[..., None].astype(np.float32), 'omap': om4, 'traj': (traj * avail).astype(np.float32), 'avail': av}
    return {k: d[k] for k in keys}
