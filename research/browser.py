"""Python port of the game's handwriting pipeline.

Two halves, both copied from the browser code so notebook numbers match the game:

- render_strokes():   js/ui/pad.js  renderForModel()  (strokes -> 280x280 white-on-black)
- preprocess_to_mnist(): js/ai/processing.js preprocessToMNIST(roi='all', center=True, blur=False)
                       (280x280 image -> 28x28 MNIST-like image in [0, 1])
"""
import numpy as np
from PIL import Image, ImageDraw

MODEL_SIZE = 280   # side of the square image fed to the preprocessing
MODEL_BOX = 190    # longest side of the digit inside that image
MODEL_LINE = 18    # stroke width inside that image
SUPERSAMPLE = 4    # PIL draws without anti-aliasing, so draw big and shrink (canvas is anti-aliased)


def js_round(x):
    """Math.round: halves go up (towards +inf), unlike numpy's round-half-to-even."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


# ----------------------------------------------------------------------------- pad.js

def render_strokes(strokes, pad_size=None, line=MODEL_LINE, box=MODEL_BOX, size=MODEL_SIZE):
    """Render strokes like pad.renderForModel().

    strokes: list of (n, 2) arrays of pad coordinates (x right, y down).
    pad_size: (w, h) of the pad; only used for the "stray dot" check (None = skip it).
    Returns a (size, size) float32 gray image in [0, 1] (white ink on black), or None.
    """
    strokes = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes if len(s)]
    if not strokes:
        return None
    pts = np.concatenate(strokes)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    ext = max(x1 - x0, y1 - y0)
    if pad_size is not None and ext < min(pad_size) * 0.06:
        return None                                    # a stray dot
    ext = max(ext, 1e-6)
    scale = box / ext
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    ss = SUPERSAMPLE
    img = Image.new('L', (size * ss, size * ss), 0)
    draw = ImageDraw.Draw(img)
    r = line * ss / 2 - 0.5                           # PIL ellipses include both edges
    for s in strokes:
        # the model gets the raw polyline (no smoothing), round caps and joins
        p = ((s - [cx, cy]) * scale + size / 2) * ss
        if len(p) > 1:
            draw.line([tuple(q) for q in p], fill=255, width=int(round(line * ss)))
        for qx, qy in p:
            draw.ellipse((qx - r, qy - r, qx + r, qy + r), fill=255)
    img = img.reduce(ss)                               # box filter = anti-aliasing
    return np.asarray(img, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------- processing.js

def otsu(gray):
    """utils.js otsu(): 256-bin histogram, first threshold with the largest between-class variance."""
    bins = np.clip(np.floor(gray.astype(np.float64) * 255), 0, 255).astype(np.int64)
    hist = np.bincount(bins.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    i = np.arange(256)
    w_b = np.cumsum(hist)
    w_f = total - w_b
    sum_all = (i * hist).sum()
    sum_b = np.cumsum(i * hist)
    ok = (w_b > 0) & (w_f > 0)
    m_b = np.where(ok, sum_b / np.maximum(w_b, 1), 0)
    m_f = np.where(ok, (sum_all - sum_b) / np.maximum(w_f, 1), 0)
    between = np.where(ok, w_b * w_f * (m_b - m_f) ** 2, -1)
    return float(np.argmax(between)) / 255.0 if ok.any() else 0.0


def preprocess_to_mnist(gray, center=True):
    """processing.js preprocessToMNIST(roi='all', center=True, blur=False).

    gray: (H, W) float image in [0, 1] (the browser computes it from RGBA with
    0.299 R + 0.587 G + 0.114 B; our renders are already gray).
    Returns a (28, 28) float32 image in [0, 1].
    """
    gray = np.asarray(gray, dtype=np.float32)
    h_src, w_src = gray.shape

    # 2) Otsu threshold + adaptive inversion
    th = otsu(gray)
    invert = gray.mean() > 0.5
    fg = (gray >= th) != invert

    # 3) bounding box of all foreground
    ys, xs = np.nonzero(fg)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        x0, y0, x1, y1 = 0, 0, w_src - 1, h_src - 1
    w, h = x1 - x0 + 1, y1 - y0 + 1

    # 4) longest side -> 20 px, nearest neighbour on the gray crop
    scale = 20 / w if w > h else 20 / h
    new_w = max(1, int(js_round(w * scale)))
    new_h = max(1, int(js_round(h * scale)))
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    sx = np.minimum(w - 1, js_round(np.arange(new_w) / scale))
    sy = np.minimum(h - 1, js_round(np.arange(new_h) / scale))
    scaled = crop[np.ix_(sy, sx)]

    # 5) paste into the centre of 28x28
    out = np.zeros((28, 28), dtype=np.float32)
    off_x, off_y = (28 - new_w) // 2, (28 - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = scaled

    # 6) move the centre of mass to (14, 14)
    if center:
        mass = out.sum()
        if mass > 0:
            yy, xx = np.mgrid[0:28, 0:28]
            cx, cy = (out * xx).sum() / mass, (out * yy).sum() / mass
            dx, dy = int(js_round(14 - cx)), int(js_round(14 - cy))
            if dx or dy:
                out = shift_image(out, dx, dy)

    # 8) min-max normalise
    lo, hi = out.min(), out.max()
    return (out - lo) / max(1e-6, hi - lo)


def shift_image(img, dx, dy):
    """Move pixels by (dx, dy), dropping what falls off the edge."""
    out = np.zeros_like(img)
    h, w = img.shape
    src = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def strokes_to_28(strokes, pad_size=None):
    """The whole browser path: strokes -> 280x280 render -> 28x28 model input (or None)."""
    g = render_strokes(strokes, pad_size)
    return None if g is None else preprocess_to_mnist(g)
