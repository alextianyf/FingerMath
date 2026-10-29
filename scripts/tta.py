"""Test-time augmentation for the shipped model: average the predictions over a few small
shifts of the 28x28 input (one batch in the browser). Is it worth the extra compute?

    python tta.py MODEL_NAME VARIANT
"""
import sys

import numpy as np
import keras

from fingermath import datasets, models, ship
from fingermath.browser import shift_image

SHIFTS = {'none': [(0, 0)], 'shift4': [(0, 0), (1, 0), (-1, 0), (0, 1)],
          'shift5': [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]}


def main(name, variant):
    m = keras.models.load_model(datasets.CACHE / 'models' / f'{name}.keras')
    predict = models.predict_fn(m)
    _, _, x_te, y_te = datasets.load_mnist()
    skel, y_skel = ship.skeleton_sets(x_te, y_te, 'skel_test', seed=2)
    tmpl, y_tmpl, _, _ = ship.template_sets(200, 5, 'tmpl_test', seed=10)
    for label, shifts in SHIFTS.items():
        row = []
        for x, y in [(skel[variant], y_skel), (tmpl[variant], y_tmpl)]:
            p = np.mean([predict(np.stack([shift_image(im, dx, dy) for im in x])) for dx, dy in shifts], 0)
            row.append((p.argmax(1) == y).mean())
        print(f'{label:7s} pad_skel {row[0]:.2%}  pad_tmpl {row[1]:.2%}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
