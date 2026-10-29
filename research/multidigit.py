"""Multi-digit answers (2-3 digits) on the pad: synthetic data and two ways to read them.

(a) segmentation: split the writing into digits, then reuse the single-digit pipeline;
    either by grouping *strokes* (the pad has them) or by connected components of pixels.
(b) sequence model: read the whole line with a CNN + CTC, no explicit segmentation.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import synth
from browser import MODEL_LINE, MODEL_BOX, SUPERSAMPLE, preprocess_to_mnist, strokes_to_28

LINE_H = 28                 # sequence-model input: 28 x LINE_W
LINE_W = 112
BLANK = 10                  # CTC blank label


# ------------------------------------------------------------------ composing numbers

def _bbox(strokes):
    p = np.concatenate(strokes)
    return p.min(0), p.max(0)


def compose(digit_strokes, rng, gap=None):
    """Place digits left to right like a player would write them.

    digit_strokes: list (one per digit) of stroke lists. Each digit is scaled to height 1
    (with +-10% jitter); `gap` is the space between digits in units of digit height
    (negative = overlapping). Returns (strokes, owner) with owner[i] = digit index of stroke i.
    """
    out, owner, x = [], [], 0.0
    for k, strokes in enumerate(digit_strokes):
        lo, hi = _bbox(strokes)
        h = max(hi[1] - lo[1], 1e-6)
        s = rng.uniform(0.9, 1.1) / h
        dy = rng.normal(0, 0.05)
        if k:
            x += rng.uniform(-0.12, 0.45) if gap is None else gap
        for st in strokes:
            out.append((st - lo) * s + [x, dy])
            owner.append(k)
        x += (hi[0] - lo[0]) * s
    return out, np.array(owner)


def render_line(strokes, height=MODEL_BOX, line=MODEL_LINE, margin=45):
    """Like pad.renderForModel(), but the *height* is normalised (a number is wide).

    Returns a float32 gray image (white on black), height + 2*margin rows.
    """
    lo, hi = _bbox(strokes)
    scale = height / max(hi[1] - lo[1], 1e-6)
    w = int((hi[0] - lo[0]) * scale + 2 * margin)
    ss = SUPERSAMPLE
    img = Image.new('L', (w * ss, (height + 2 * margin) * ss), 0)
    draw = ImageDraw.Draw(img)
    r = line * ss / 2 - 0.5
    for s in strokes:
        p = ((s - lo) * scale + margin) * ss
        if len(p) > 1:
            draw.line([tuple(q) for q in p], fill=255, width=int(round(line * ss)))
        for qx, qy in p:
            draw.ellipse((qx - r, qy - r, qx + r, qy + r), fill=255)
    return np.asarray(img.reduce(ss), dtype=np.float32) / 255.0


def to_line_input(gray):
    """Rendered line -> 28 x 112 sequence-model input (ink height 20 px, centred)."""
    ys, xs = np.nonzero(gray > 0.5)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = min(20 / h, (LINE_W - 8) / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BOX),
                       np.float32) / 255
    out = np.zeros((LINE_H, LINE_W), np.float32)
    oy, ox = (LINE_H - nh) // 2, (LINE_W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = small
    return out


# ------------------------------------------------------------------ (a) segmentation

def group_strokes(strokes, min_overlap=0.3, min_width=0.15):
    """Group strokes into digits by horizontal overlap (works on raw pad strokes).

    Strokes are taken left to right; a stroke joins the current group when their x-ranges
    overlap by more than `min_overlap` of the narrower one. A vertical '1' has ~zero width,
    so widths are floored at `min_width` x the line height.
    """
    lo, hi = _bbox(strokes)
    height = max(hi[1] - lo[1], 1e-6)
    spans = [(s[:, 0].min(), s[:, 0].max()) for s in strokes]
    order = np.argsort([(a + b) / 2 for a, b in spans])
    groups, g_span = [], None
    for i in order:
        a, b = spans[i]
        c = (a + b) / 2
        a, b = min(a, c - min_width * height / 2), max(b, c + min_width * height / 2)
        if groups:
            ga, gb = g_span
            overlap = min(gb, b) - max(ga, a)
            if overlap > min_overlap * min(gb - ga, b - a):
                groups[-1].append(i)
                g_span = (min(ga, a), max(gb, b))
                continue
        groups.append([i])
        g_span = (a, b)
    return groups


def read_by_strokes(strokes, predict):
    """Stroke grouping -> one 28x28 per group (browser pipeline) -> classifier."""
    groups = group_strokes(strokes)
    imgs = np.stack([strokes_to_28([strokes[i] for i in g]) for g in groups])
    return ''.join(str(d) for d in predict(imgs).argmax(1))


def components(gray, min_overlap=0.5, min_area=0.01):
    """Connected components of the ink, merged when they overlap horizontally
    (a 5's top bar or a crossed 7 are separate components). Returns a list of masks."""
    ink = gray > 0.5
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    if n == 0:
        return []
    sizes = ndimage.sum(ink, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] >= min_area * ink.sum()]
    sl = ndimage.find_objects(lab)
    items = sorted(((sl[i - 1][1].start, sl[i - 1][1].stop, i) for i in keep))
    groups = []
    for a, b, i in items:
        if groups:
            ga, gb, ids = groups[-1]
            if min(gb, b) - max(ga, a) > min_overlap * min(gb - ga, b - a):
                groups[-1] = (min(ga, a), max(gb, b), ids + [i])
                continue
        groups.append((a, b, [i]))
    return [np.isin(lab, ids) for _, _, ids in groups]


def read_by_components(gray, predict):
    masks = components(gray)
    if not masks:
        return ''
    # keep the anti-aliased edge around each component: on a hard 0/1 image the browser's Otsu
    # threshold becomes 0 and the whole canvas would count as ink (see README)
    imgs = np.stack([preprocess_to_mnist(gray * ndimage.binary_dilation(m, iterations=2)) for m in masks])
    return ''.join(str(d) for d in predict(imgs).argmax(1))


# ------------------------------------------------------------------ (b) CTC decoding

def ctc_greedy(scores):
    """scores (T, 11), logits or probabilities -> string: best class per step, collapse repeats, drop blanks."""
    best = scores.argmax(-1)
    out, prev = [], BLANK
    for c in best:
        if c != prev and c != BLANK:
            out.append(str(c))
        prev = c
    return ''.join(out)


# ------------------------------------------------------------------ data generation (picklable)

def _skeleton_digit(img, rng):
    return synth.skeleton_strokes(img, rng)


def _make_sequences_job(args):
    """Build sequences from MNIST skeletons or template writers.

    args = (kind, images, labels, n, seed, lengths, keep); kind is 'skel' (MNIST skeletons,
    images/labels given) or 'tmpl' (template writers, 5 numbers per writer).
    Returns (line inputs, labels as strings, gaps, strokes); strokes only when keep=True
    (the segmentation methods need them; md.render_line(strokes) redraws the image).
    """
    kind, images, labels, n, seed, lengths, keep = args
    rng = np.random.default_rng(seed)
    xs, ys, gaps, stroke_sets = [], [], [], []
    writer = synth.make_writer(rng)
    for j in range(n):
        k = int(rng.choice(lengths))
        gap = rng.uniform(-0.12, 0.45)
        if kind == 'skel':
            idx = rng.integers(len(images), size=k)
            digits = [int(labels[i]) for i in idx]
            parts = [_skeleton_digit(images[i], rng) for i in idx]
        else:
            if j % 5 == 0:
                writer = synth.make_writer(rng)
            digits = [int(d) for d in rng.integers(10, size=k)]
            if k > 1 and digits[0] == 0:
                digits[0] = int(rng.integers(1, 10))       # numbers don't start with 0
            parts = [synth.template_strokes(d, writer, rng) for d in digits]
        strokes, _ = compose(parts, rng, gap)
        g = render_line(strokes)
        xs.append(to_line_input(g))
        ys.append(''.join(map(str, digits)))
        gaps.append(gap)
        if keep:
            stroke_sets.append(strokes)
    return np.stack(xs), ys, np.array(gaps), stroke_sets
