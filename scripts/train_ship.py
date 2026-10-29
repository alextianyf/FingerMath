"""Notebook-06 training runs: which pipeline variant and data give the best shipped digit model.

Every run trains the notebook-03 CNN with the notebook-03 budget (8 epochs x 470 steps of 128,
Adam + cosine decay), half MNIST (augmented) and half MNIST skeletons redrawn with the pad's pen,
both seen through the given pipeline variant. `pad_tmpl` (template writers) is never trained on.

    python train_ship.py            # all runs (cached in cache/models/ship_*.keras)
    python train_ship.py final      # + the final model (longer, label smoothing)

Writes results/06_single.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

from fingermath import datasets, models, ship
from fingermath import evaluate as ev

OUT = Path(__file__).resolve().parents[1] / 'results'
MODEL_DIR = datasets.CACHE / 'models'
PAIRS = [(1, 7), (4, 9), (5, 6), (3, 5), (2, 7), (3, 8), (0, 6), (1, 2)]

# run -> (pipeline variant, pad data key, epochs, label smoothing)
RUNS = {
    'cnn_nn18': ('nn18', 'nn18', 8, 0.0),
    'cnn_area18': ('area18', 'area18', 8, 0.0),
    'cnn_area18_pen': ('area18', 'area18_penaug', 8, 0.0),
    'cnn_area26_pen': ('area26', 'area26_penaug', 8, 0.0),
}
FINAL = ('cnn_final', None, 16, 0.1)      # variant chosen from the runs above


def train(name, pad_key, epochs, smoothing, x_mn, y_mn, pad, y_pad):
    path = MODEL_DIR / f'ship_{name}.keras'
    if path.exists():
        return keras.models.load_model(path), None
    keras.utils.set_random_seed(42)
    m = models.build_cnn(name)
    ds = models.make_train_ds([(x_mn, y_mn, models.augment_mnist, 0.5), (pad[pad_key], y_pad, models.augment_light, 0.5)])
    if smoothing:   # one-hot labels for label smoothing
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, 10)))
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
    else:
        loss = 'sparse_categorical_crossentropy'
    sched = keras.optimizers.schedules.CosineDecay(2e-3, 470 * epochs)
    m.compile(optimizer=keras.optimizers.Adam(sched), loss=loss)
    t0 = time.time()
    m.fit(ds, steps_per_epoch=470, epochs=epochs, verbose=0)
    m.save(path)
    return m, time.time() - t0


def report(probs, y):
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == y
    pairs = {}
    for a, b in PAIRS:
        n = ((y == a) | (y == b)).sum()
        pairs[f'{a}-{b}'] = float((((y == a) & (pred == b)) | ((y == b) & (pred == a))).sum() / n)
    hit, wrong, _ = ev.game_hit_rate(probs, y, 2)
    return {
        'acc': float(ok.mean()),
        'per_digit': [float(ok[y == d].mean()) for d in range(10)],
        'pairs': pairs,
        'wrong_confident': float((~ok & (conf > 0.9)).mean()),
        'game_hit_2': float(hit), 'game_wrong_2': float(wrong),
    }


def main():
    x_tr, y_tr, x_te, y_te = datasets.load_mnist()
    x_mn = x_tr.astype(np.float32) / 255
    pad, y_pad = ship.skeleton_sets(x_tr, y_tr, 'skel_train', seed=1, pens=True)
    skel, y_skel = ship.skeleton_sets(x_te, y_te, 'skel_test', seed=2)
    tmpl, y_tmpl, _, _ = ship.template_sets(200, 5, 'tmpl_test', seed=10)
    mnist = x_te.astype(np.float32) / 255

    def evaluate(predict, variant):
        return {'mnist': report(predict(mnist), y_te),
                'pad_skel': report(predict(skel[variant]), y_skel),
                'pad_tmpl': report(predict(tmpl[variant]), y_tmpl)}

    results = {}
    mlp = tf.saved_model.load(str(Path(__file__).resolve().parents[1] / 'baselines' / 'mlp' / 'saved_model')).signatures['serving_default']
    mlp_predict = lambda x: np.concatenate([mlp(keras_tensor=tf.constant(x[i:i + 2048].reshape(-1, 784)))['output_0'].numpy()
                                            for i in range(0, len(x), 2048)])
    results['mlp_deployed'] = {'variant': 'nn18', 'params': 435_404, **evaluate(mlp_predict, 'nn18')}

    runs = dict(RUNS)
    if 'final' in sys.argv:
        best = max(RUNS, key=lambda r: json.loads((OUT / '06_single.json').read_text())[r]['pad_skel']['acc'])  # pad_tmpl stays the test set
        variant, pad_key = RUNS[best][:2]
        runs[FINAL[0]] = (variant, pad_key, FINAL[2], FINAL[3])
    for name, (variant, pad_key, epochs, smoothing) in runs.items():
        m, secs = train(name, pad_key, epochs, smoothing, x_mn, y_tr, pad, y_pad)
        results[name] = {'variant': variant, 'pad_data': pad_key, 'epochs': epochs, 'label_smoothing': smoothing,
                         'params': m.count_params(), **evaluate(models.predict_fn(m), variant)}
        if secs:
            results[name]['train_minutes'] = secs / 60
        r = results[name]
        print(f"{name:16s} mnist {r['mnist']['acc']:.2%}  pad_skel {r['pad_skel']['acc']:.2%}  "
              f"pad_tmpl {r['pad_tmpl']['acc']:.2%}", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / '06_single.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
