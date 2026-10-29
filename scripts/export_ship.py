"""Export the shipped digit model to TF.js (graph model, float16 weights) into /digit_model,
plus a few test inputs with the expected outputs (export/ship_check.json) for export/bench.html.

    python export_ship.py MODEL_NAME VARIANT        e.g. ship_cnn_final area18
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import keras
from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs_conv

from fingermath import datasets, ship

ROOT = Path(__file__).resolve().parents[1]
# the game repo, checked out next to this one (override with MATH_RUN=...)
TARGET = Path(os.environ.get('MATH_RUN', ROOT.parent / 'Math-Run')) / 'digit_model'


def main(name, variant):
    model = keras.models.load_model(datasets.CACHE / 'models' / f'{name}.keras')
    saved = datasets.CACHE / f'saved_{name}'
    model.export(str(saved), verbose=False)
    shutil.rmtree(TARGET, ignore_errors=True)
    tfjs_conv.convert_tf_saved_model(str(saved), str(TARGET), signature_def='serving_default',
                                     saved_model_tags='serve', quantization_dtype_map={'float16': '*'})
    print('digit_model:', sum(f.stat().st_size for f in TARGET.iterdir()) // 1024, 'KB')

    sets, y, _, _ = ship.template_sets(200, 5, 'tmpl_test', seed=10)
    x = sets[variant][:32]
    p = model.predict(x[..., None], verbose=0)
    (ROOT / 'exports' / 'ship_check.json').write_text(json.dumps({
        'model': name, 'variant': variant, 'x': np.round(x, 4).reshape(32, -1).tolist(), 'y': y[:32].tolist(),
        'probs': np.round(p, 5).tolist()}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
