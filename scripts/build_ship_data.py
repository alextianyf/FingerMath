"""Build the notebook-06 datasets (cache/ship_*.npz). Takes ~10 minutes on 12 cores."""
import time
import datasets, ship

if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te = datasets.load_mnist()
    t = time.time()
    ship.template_sets(200, 5, 'tmpl_test', seed=10); print('tmpl_test', time.time() - t, flush=True)
    ship.skeleton_sets(x_te, y_te, 'skel_test', seed=2); print('skel_test', time.time() - t, flush=True)
    ship.skeleton_sets(x_tr, y_tr, 'skel_train', seed=1, pens=True); print('skel_train', time.time() - t, flush=True)
