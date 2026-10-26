"""Evaluation helpers shared by the notebooks."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import datasets


def load_test_sets(with_real=True):
    """name -> (x (N, 28, 28) float in [0, 1], y). Builds nothing: run notebook 01 first."""
    _, _, x_test, y_test = datasets.load_mnist()
    sets = {
        'mnist': (x_test.astype(np.float32) / 255, y_test),
        'mnist_pipe': (datasets.mnist_through_pipeline(x_test, 'mnist_test_pipe'), y_test),
        'pad_skel': datasets.pad_skeleton_set(x_test, y_test, 'pad_skel_test', seed=2),
        'pad_tmpl': datasets.template_set(200, 5, 'pad_tmpl_test', seed=10)[:2],
    }
    real = [s for s in datasets.load_real_samples() if s['writer'] != 'pipeline-check'] if with_real else []
    if real:
        sets['real'] = (np.stack([s['img28'] for s in real]).astype(np.float32),
                        np.array([int(s['label']) for s in real]))
    return sets


def accuracy_table(predict, sets):
    """predict: x (N, 28, 28) -> probs (N, 10). Returns {name: accuracy}."""
    return {name: float((predict(x).argmax(1) == y).mean()) for name, (x, y) in sets.items()}


def decide(probs, answers):
    """js/game/handwriting.js decide(): the game reads the digit together with the answers on screen."""
    order = np.argsort(-probs)
    top = order[0]
    best = next(d for d in order if d in answers)
    if best == top and probs[top] >= 0.3:
        return 'hit', top
    if probs[best] >= 0.2 and probs[best] >= 0.35 * probs[top]:
        return 'hit', best
    if probs[top] >= 0.5:
        return 'miss', top
    return 'unclear', top


def game_hit_rate(probs, y, n_answers=3, seed=0):
    """How often the game accepts the written answer when `n_answers` distinct answers
    (the true digit among them) are on screen. Returns (hit, wrong-target, unclear/miss)."""
    rng = np.random.default_rng(seed)
    right = wrong = 0
    for p, t in zip(probs, y):
        others = rng.choice([d for d in range(10) if d != t], n_answers - 1, replace=False)
        kind, d = decide(p, set(others) | {t})
        right += kind == 'hit' and d == t
        wrong += kind == 'hit' and d != t
    n = len(y)
    return right / n, wrong / n, 1 - (right + wrong) / n


def plot_confusion(y, pred, title, ax=None):
    cm = confusion_matrix(y, pred, labels=range(10))
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax = ax or plt.gca()
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f'{cm[i, j] * 100:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, fontsize=10)


def show_errors(x, y, pred, probs, n=16, title=''):
    idx = np.nonzero(pred != y)[0][:n]
    fig, ax = plt.subplots(1, len(idx), figsize=(len(idx) * 0.9, 1.4))
    for a, i in zip(np.atleast_1d(ax), idx):
        a.imshow(x[i], cmap='gray'); a.axis('off')
        a.set_title(f'{y[i]}→{pred[i]}\n{probs[i].max():.0%}', fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha='left'); plt.tight_layout(); plt.show()
