"""Synthetic "pad-like" handwriting.

The game never sees scanned pen strokes like MNIST: it sees polylines from a mouse,
finger or stylus, re-rendered with a round pen of fixed width (pad.js). Two generators
produce that kind of input before real samples exist:

1. skeleton_strokes(): take an MNIST digit, thin it to a 1-px skeleton and redraw the
   skeleton with the pad's pen. Same shapes as MNIST, but the game's rendering.
2. template_strokes(): hand-made stroke templates (several styles per digit) with
   writer-level and sample-level distortions. Shapes that MNIST never saw; also gives
   us "writers" for the personalisation experiment.

Both return strokes in arbitrary units; browser.render_strokes() normalises the size.
"""
import numpy as np
from scipy.ndimage import zoom
from skimage.morphology import skeletonize

from .browser import strokes_to_28


# ------------------------------------------------------------------ distortions

def affine(pts, rot_deg=0.0, shear=0.0, sx=1.0, sy=1.0):
    """Rotate/shear/scale points around their centre."""
    c = pts.mean(0)
    t = np.radians(rot_deg)
    m = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ \
        np.array([[1.0, -shear], [0.0, 1.0]]) @ np.diag([sx, sy])
    return (pts - c) @ m.T + c


def smooth_warp(pts, rng, amp, n_waves=3):
    """Low-frequency displacement field (a cheap elastic distortion)."""
    out = pts.copy()
    for _ in range(n_waves):
        f = rng.uniform(0.5, 1.6, size=2) * rng.choice([-1, 1], size=2)
        phase = rng.uniform(0, 2 * np.pi, size=2)
        a = rng.normal(0, amp, size=2)
        arg = 2 * np.pi * (pts @ f)
        out[:, 0] += a[0] * np.sin(arg + phase[0])
        out[:, 1] += a[1] * np.sin(arg + phase[1])
    return out


def _apply(strokes, fn):
    """Apply a point transform to all strokes at once (keeps their shared geometry)."""
    lens = [len(s) for s in strokes]
    pts = fn(np.concatenate(strokes))
    return np.split(pts, np.cumsum(lens)[:-1])


# ------------------------------------------------------------- MNIST skeletons

def mnist_skeleton(img28, up=4):
    """Skeleton pixels of an MNIST digit, as (n, 2) points in 28-px units."""
    big = zoom(img28.astype(np.float32) / 255.0, up, order=1)
    sk = skeletonize(big > 0.4)
    ys, xs = np.nonzero(sk)
    if len(xs) < 2:                    # too faint: fall back to the thresholded pixels
        ys, xs = np.nonzero(big > 0.4)
    return np.stack([xs, ys], 1).astype(np.float64) / up


def skeleton_strokes(img28, rng, jitter=True):
    """MNIST digit -> skeleton -> one dot per skeleton pixel (they overlap into a pen stroke)."""
    pts = mnist_skeleton(img28)
    if jitter:
        pts = affine(pts, rng.normal(0, 6), rng.normal(0, 0.1),
                     rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1))
        pts = smooth_warp(pts / 20, rng, 0.02) * 20
    return [p[None] for p in pts]


# --------------------------------------------------------------- stroke templates

def arc(cx, cy, rx, ry, a0, a1, n=None):
    """Elliptic arc; angles in degrees, y points down (so increasing angle = clockwise)."""
    n = n or max(8, int(abs(a1 - a0) / 10))
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def line(*pts, step=0.04):
    """Polyline through the given points, resampled every `step` units."""
    pts = np.asarray(pts, dtype=np.float64)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * (i / k) for i in range(1, k + 1)]
    return np.array(out)


def cat(*parts):
    return np.concatenate(parts)


# Each digit: a list of styles; each style returns a list of strokes in a ~0.6 x 1 box.
TEMPLATES = {
    0: [lambda: [arc(.3, .5, .28, .48, -90, 275)],
        lambda: [arc(.3, .5, .26, .48, -80, -445)]],
    1: [lambda: [line((.3, 0), (.3, 1))],
        lambda: [line((.1, .2), (.3, 0), (.3, 1))],
        lambda: [line((.1, .2), (.3, 0), (.3, 1)), line((.08, 1), (.52, 1))]],
    2: [lambda: [cat(arc(.3, .28, .27, .27, 200, 390), line((.55, .38), (0, 1), (.6, 1)))],
        lambda: [cat(arc(.3, .3, .28, .3, 190, 360), arc(.0, .3, .58, .7, 0, 90),
                     line((0, 1), (.65, .98)))]],
    3: [lambda: [cat(arc(.3, .25, .26, .24, 200, 450), arc(.3, .74, .3, .26, 270, 515))],
        lambda: [cat(line((0, 0), (.55, 0), (.22, .4)), arc(.27, .7, .32, .3, 260, 510))]],
    4: [lambda: [line((.15, 0), (0, .62), (.62, .62)), line((.45, .25), (.45, 1))],
        lambda: [line((.45, 1), (.45, 0), (0, .66), (.62, .66))]],
    5: [lambda: [cat(line((.1, 0), (.06, .44)), arc(.28, .68, .29, .3, 225, 500)), line((.1, 0), (.55, 0))],
        lambda: [cat(line((.55, 0), (.08, 0), (.05, .44)), arc(.28, .68, .29, .3, 225, 500))]],
    6: [lambda: [cat(line((.5, 0), (.25, .2), (.05, .55)), arc(.3, .72, .27, .27, 190, -170))],
        lambda: [cat(arc(.75, .7, .72, .7, 240, 180), arc(.3, .72, .27, .27, 180, -175))]],
    7: [lambda: [line((0, 0), (.6, 0), (.2, 1))],
        lambda: [line((0, 0), (.6, 0), (.2, 1)), line((.15, .5), (.55, .5))],
        lambda: [line((0, .14), (0, 0), (.6, 0), (.3, 1))]],
    8: [lambda: [cat(arc(.3, .25, .22, .23, 90, 450), arc(.3, .73, .28, .27, -90, -450))],
        lambda: [cat(arc(.3, .25, .23, .24, 30, -200), line((.08, .3), (.55, .68)),
                     arc(.3, .73, .26, .26, -30, 200), line((.06, .7), (.5, .32)))]],
    9: [lambda: [cat(arc(.3, .27, .27, .26, 0, -360), line((.57, .27), (.55, 1)))],
        lambda: [cat(arc(.3, .27, .27, .26, 0, -360), arc(-.2, .27, .77, .73, 0, 90))]],
}


def make_writer(rng, noise=1.0, switch=0.0):
    """A writer = a style choice per digit + a consistent slant/width/shape quirk.

    noise scales the sample-to-sample variation; switch is the chance that a sample uses
    another style of the digit (people are not perfectly consistent).
    """
    return {
        'noise': noise,
        'switch': switch,
        'style': {d: int(rng.integers(len(TEMPLATES[d]))) for d in range(10)},
        'slant': rng.normal(0, 0.18),
        'rot': rng.normal(0, 4),
        'aspect': rng.uniform(0.8, 1.2),
        'quirk_seed': int(rng.integers(1 << 31)),   # writer-specific warp of each digit
        'quirk_amp': rng.uniform(0.015, 0.04),
        'tremor': rng.uniform(0.002, 0.008),        # mouse users shake more than pen users
    }


def template_strokes(digit, writer, rng):
    """One sample of `digit` written by `writer`."""
    style = writer['style'][digit]
    if writer.get('switch', 0) and rng.random() < writer['switch']:
        style = int(rng.integers(len(TEMPLATES[digit])))
    strokes = TEMPLATES[digit][style]()
    n = writer.get('noise', 1.0)
    quirk_rng = np.random.default_rng(writer['quirk_seed'] + digit)
    strokes = _apply(strokes, lambda p: smooth_warp(p, quirk_rng, writer['quirk_amp']))
    strokes = _apply(strokes, lambda p: affine(
        p, writer['rot'] + rng.normal(0, 4 * n), writer['slant'] + rng.normal(0, 0.06 * n),
        writer['aspect'] * rng.uniform(1 - 0.1 * n, 1 + 0.1 * n), 1.0))
    strokes = _apply(strokes, lambda p: smooth_warp(p, rng, 0.015 * n))
    # separate strokes of one digit rarely land exactly where they should
    strokes = [s + rng.normal(0, 0.025 * n, size=2) if i else s for i, s in enumerate(strokes)]
    return [s + rng.normal(0, writer['tremor'], size=s.shape) for s in strokes]


# ------------------------------------------------------------ batch helpers (picklable)

def _skeleton_job(args):
    imgs, seed, jitter = args
    rng = np.random.default_rng(seed)
    return np.stack([strokes_to_28(skeleton_strokes(im, rng, jitter)) for im in imgs])


def _template_job(args):
    digits, writer, seed = args
    rng = np.random.default_rng(seed)
    return np.stack([strokes_to_28(template_strokes(d, writer, rng)) for d in digits])
