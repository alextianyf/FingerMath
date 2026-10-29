"""Models, augmentation and training loop shared by notebooks 03-05."""
import numpy as np
import tensorflow as tf
import keras
from keras import layers


def build_mlp(name='mlp'):
    """Same architecture as the deployed model (mlp_training/notebook.ipynb)."""
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name=name)


def build_cnn(name='cnn', width=32):
    """Small VGG-style CNN (~107k parameters with width=32, ~27k with width=16).

    Global average pooling instead of a big flatten keeps the parameter count low; the
    64-d 'embedding' layer is what notebook 04 fine-tunes on top of.
    """
    w = width
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(w, 3, padding='same', activation='relu'),
        layers.Conv2D(w, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                                   # 14x14
        layers.Conv2D(2 * w, 3, padding='same', activation='relu'),
        layers.Conv2D(2 * w, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                                   # 7x7
        layers.Conv2D(2 * w, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', name='embedding'),
        layers.Dense(10, activation='softmax', name='classifier'),
    ], name=name)


# ------------------------------------------------------------------ augmentation

_affine = keras.Sequential([
    layers.RandomRotation(0.045, fill_mode='constant'),         # about +-16 degrees
    layers.RandomZoom((-0.15, 0.15), fill_mode='constant'),
    layers.RandomTranslation(0.08, 0.08, fill_mode='constant'),
], name='affine')


def _thickness(x):
    """Per-image random stroke thickness: erode (thinner), keep, or dilate (thicker)."""
    dil = tf.nn.max_pool2d(x, 3, 1, 'SAME')
    ero = -tf.nn.max_pool2d(-x, 2, 1, 'SAME')        # 2x2 erosion: removes ~1 px of a stroke
    pick = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0, 3, dtype=tf.int32)
    return tf.where(pick == 0, ero, tf.where(pick == 1, x, dil))


def _sharpen(x):
    """Sometimes push gray edges towards black/white like the pad's near-binary images."""
    t = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.2, 0.6)
    hard = tf.cast(x > t, x.dtype)
    use = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5, x.dtype)
    return use * hard + (1 - use) * x


def augment_mnist(x):
    x = _affine(x, training=True)
    x = _thickness(x)
    x = _sharpen(x)
    return tf.clip_by_value(x, 0, 1)


def augment_light(x):
    return tf.clip_by_value(_affine(x, training=True), 0, 1)


def make_train_ds(sources, batch=128, seed=0):
    """sources: list of (x, y, augment_fn or None, fraction of each batch).

    Every batch holds a fixed share of each source, each augmented its own way.
    """
    parts = []
    for i, (x, y, aug, frac) in enumerate(sources):
        ds = tf.data.Dataset.from_tensor_slices((x[..., None].astype(np.float32), y.astype(np.int32)))
        ds = ds.shuffle(len(x), seed=seed + i).repeat().batch(max(1, round(batch * frac)))
        if aug is not None:
            ds = ds.map(lambda a, b, f=aug: (f(a), b), num_parallel_calls=tf.data.AUTOTUNE)
        parts.append(ds)
    ds = tf.data.Dataset.zip(tuple(parts)).map(
        lambda *p: (tf.concat([a for a, _ in p], 0), tf.concat([b for _, b in p], 0)))
    return ds.prefetch(tf.data.AUTOTUNE)


def train(model, ds, steps_per_epoch, epochs, lr=2e-3, verbose=2):
    sched = keras.optimizers.schedules.CosineDecay(lr, steps_per_epoch * epochs)
    model.compile(optimizer=keras.optimizers.Adam(sched),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)


def predict_fn(model, batch=1024):
    return lambda x: model.predict(x[..., None].astype(np.float32), batch_size=batch, verbose=0)
