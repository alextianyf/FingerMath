"""Fixtures for e2e/index.html: 100 samples of each digit from the notebook-06 pad_tmpl test set
(template writers, the generator no model is trained on), as raw strokes in pad pixels.
"""
import json
from pathlib import Path

import numpy as np

import ship

PX = 150          # digit height in pad pixels

if __name__ == '__main__':
    _, y, _, strokes = ship.template_sets(200, 5, 'tmpl_test', seed=10)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.nonzero(y == d)[0], 100, replace=False) for d in range(10)])
    samples = [{'label': int(y[i]), 'strokes': [np.round(np.asarray(s) * PX).astype(int).tolist() for s in strokes[i]]}
               for i in idx]
    path = Path(__file__).parent / 'e2e' / 'fixtures.json'
    path.write_text(json.dumps({'samples': samples}, separators=(',', ':')))
    print(path, path.stat().st_size // 1024, 'KB')
