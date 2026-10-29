"""Multi-digit approaches compared on the same numbers as e2e/fixtures.json (template writers,
gap between digits uniform in [-0.12, 0.45] digit heights), with the shipped digit CNN.

- oracle:     each digit cut out with its true strokes. What writing in separate boxes, or pausing
              after every digit, would give (segmentation is then free) - an upper bound;
- strokes:    stroke grouping by horizontal overlap (multidigit.group_strokes, = js/ai/segment.js
              without its speck rule and the alternative readings);
- components: connected components of the rendered ink, merged by horizontal overlap.

    python compare_multi.py MODEL_NAME VARIANT     e.g. ship_cnn_final area18

Writes results/06_multi_python.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import keras
from scipy import ndimage

from fingermath import datasets, models, ship, synth
from fingermath import multidigit as md

ROOT = Path(__file__).resolve().parents[1]


def numbers_with_owner(n, length, seed):
    """export_e2e.numbers() (same generator and seeds), keeping which digit each stroke belongs to."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        if j % 5 == 0:
            writer = synth.make_writer(rng)
        digits = [int(d) for d in rng.integers(10, size=length)]
        digits[0] = digits[0] or int(rng.integers(1, 10))
        gap = float(rng.uniform(-0.12, 0.45))
        strokes, owner = md.compose([synth.template_strokes(d, writer, rng) for d in digits], rng, gap)
        out.append((''.join(map(str, digits)), gap, strokes, owner))
    return out


def main(model_name, variant):
    line, resample = ship.VARIANTS[variant]
    model = keras.models.load_model(datasets.CACHE / 'models' / f'{model_name}.keras')
    predict = models.predict_fn(model)
    read = lambda imgs: ''.join(str(d) for d in predict(np.stack(imgs)).argmax(1))
    to28 = lambda strokes: ship.to28(strokes, line, resample)

    def by_components(strokes):
        gray = md.render_line(strokes, line=line)
        masks = md.components(gray)
        if not masks:
            return ''
        return read([ship.preprocess(gray * ndimage.binary_dilation(m, iterations=2), resample) for m in masks])

    results = {}
    for name, (n, length, seed) in {'multi2': (500, 2, 1), 'multi3': (300, 3, 2)}.items():
        rows = numbers_with_owner(n, length, seed)
        hits = {'oracle': [], 'strokes': [], 'components': []}
        gaps = []
        for label, gap, strokes, owner in rows:
            gaps.append(gap)
            hits['oracle'].append(read([to28([s for s, o in zip(strokes, owner) if o == k]) for k in range(length)]) == label)
            groups = md.group_strokes(strokes)
            hits['strokes'].append(read([to28([strokes[i] for i in g]) for g in groups]) == label)
            hits['components'].append(by_components(strokes) == label)
        gaps = np.array(gaps)
        bins = {'overlap (<0)': gaps < 0, 'tight (0-0.1)': (gaps >= 0) & (gaps < 0.1),
                'normal (0.1-0.25)': (gaps >= 0.1) & (gaps < 0.25), 'wide (>0.25)': gaps >= 0.25}
        results[name] = {k: {'all': float(np.mean(v)), **{b: float(np.mean(np.array(v)[m])) for b, m in bins.items()}}
                         for k, v in hits.items()}
        print(name, json.dumps(results[name], indent=1), flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / '06_multi_python.json').write_text(json.dumps({'model': model_name, 'variant': variant, **results}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
