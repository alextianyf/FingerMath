"""Notebook 12: image-model architectures compared on notebook 09's data (config C).

Every model takes a (28, 28, 1) image, ends in a 64-d 'embedding' layer and a 10-way softmax
'classifier', so they all plug into the same training, evaluation and personalisation code.
"""
import keras
from keras import layers

import models


def _head(h, name):
    h = layers.GlobalAveragePooling2D()(h) if len(h.shape) == 4 else h
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(64, activation='relu', name='embedding')(h)
    return layers.Dense(10, activation='softmax', name='classifier')(h)


def _conv_bn(h, f, k=3, stride=1):
    h = layers.Conv2D(f, k, strides=stride, padding='same', use_bias=False)(h)
    return layers.ReLU()(layers.BatchNormalization()(h))


def resnet(name='resnet'):
    """Three stages (32, 48, 64 channels) of one residual block each [25]; strided conv between stages."""
    inp = layers.Input((28, 28, 1))
    h = _conv_bn(inp, 32)
    for i, f in enumerate([32, 48, 64]):
        if i:
            h = _conv_bn(h, f, stride=2)                      # 28 -> 14 -> 7
        r = _conv_bn(h, f)
        r = layers.BatchNormalization()(layers.Conv2D(f, 3, padding='same', use_bias=False)(r))
        h = layers.ReLU()(layers.Add()([h, r]))
    return keras.Model(inp, _head(h, name), name=name)


def separable(name='separable'):
    """MobileNet-style [26]: two plain convs at 28x28, then depthwise-separable convs (3x3 per channel
    + 1x1 across channels) at 14x14 and 7x7. (Depthwise convs are slow in TensorFlow on CPU, so the
    full-resolution layers stay plain.)"""
    inp = layers.Input((28, 28, 1))
    h = layers.MaxPooling2D()(_conv_bn(_conv_bn(inp, 32), 32))      # 14 x 14
    for f, pool in [(64, False), (64, True), (128, False), (128, False)]:
        h = layers.SeparableConv2D(f, 3, padding='same', use_bias=False)(h)
        h = layers.ReLU()(layers.BatchNormalization()(h))
        if pool:
            h = layers.MaxPooling2D()(h)                            # 7 x 7
    return keras.Model(inp, _head(h, name), name=name)


def convmixer(name='convmixer', dim=128, depth=6, patch=4, kernel=5):
    """ConvMixer [30]: patch embedding, then depth x (residual depthwise conv, pointwise conv)."""
    inp = layers.Input((28, 28, 1))
    h = layers.Conv2D(dim, patch, strides=patch)(inp)               # 7 x 7 tokens
    h = layers.BatchNormalization()(layers.Activation('gelu')(h))
    for _ in range(depth):
        r = layers.DepthwiseConv2D(kernel, padding='same')(h)
        r = layers.BatchNormalization()(layers.Activation('gelu')(r))
        h = layers.Add()([h, r])
        h = layers.Conv2D(dim, 1)(h)
        h = layers.BatchNormalization()(layers.Activation('gelu')(h))
    return keras.Model(inp, _head(h, name), name=name)


BUILDERS = {
    'cnn': lambda: models.build_cnn('cnn'),                        # the shipped model (notebooks 06, 09)
    'cnn_wide': lambda: models.build_cnn('cnn_wide', width=64),
    'resnet': resnet,
    'separable': separable,
    'convmixer': convmixer,
}


# ------------------------------------------------------------------ notebook 12b: image + pen path
# Inputs (by name): 'img' (28, 28, 1) the game's image; 'omap' (28, 28, 4) the order map
# (dir_x, dir_y, order, available) from multiview.py; 'traj' (64, 5) the pen path from
# trajectory.py; 'avail' (1,) 1 when the pen path is known. Image-only training samples (MNIST)
# come with zeros and avail = 0, so every model also learns to read a plain image.

def _cnn_trunk(h, w=32):
    """build_cnn's convolutions: returns the 7x7 feature map."""
    h = layers.Conv2D(w, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(w, 3, padding='same', activation='relu')(h)
    h = layers.MaxPooling2D()(h)
    h = layers.Conv2D(2 * w, 3, padding='same', activation='relu')(h)
    h = layers.Conv2D(2 * w, 3, padding='same', activation='relu')(h)
    h = layers.MaxPooling2D()(h)
    return layers.Conv2D(2 * w, 3, padding='same', activation='relu')(h)


def _traj_trunk(t):
    """notebook 11's 1D CNN: returns 16 x 96 path features."""
    for f in [48, 48]:
        t = layers.Conv1D(f, 5, padding='same', activation='relu')(t)
    t = layers.MaxPooling1D()(t)
    for f in [96, 96]:
        t = layers.Conv1D(f, 5, padding='same', activation='relu')(t)
    t = layers.MaxPooling1D()(t)
    return layers.Conv1D(96, 3, padding='same', activation='relu')(t)


def ordermap_cnn(name='ordermap_cnn'):
    """One image CNN on 5 channels: ink + where the pen went and in which order (ours)."""
    img, om = layers.Input((28, 28, 1), name='img'), layers.Input((28, 28, 4), name='omap')
    h = _cnn_trunk(layers.Concatenate()([img, om]))
    return keras.Model({'img': img, 'omap': om}, _head(h, name), name=name)


def dual_stream(name='dual_stream'):
    """Image CNN and path 1D CNN trained together; their pooled features meet in one head."""
    img, tr, av = layers.Input((28, 28, 1), name='img'), layers.Input((64, 5), name='traj'), layers.Input((1,), name='avail')
    fi = layers.GlobalAveragePooling2D()(_cnn_trunk(img))
    ft = layers.GlobalAveragePooling1D()(_traj_trunk(tr))
    h = layers.Concatenate()([fi, ft, av])
    return keras.Model({'img': img, 'traj': tr, 'avail': av}, _head(h, name), name=name)


def ink_glance(name='ink_glance'):
    """The pen path looks at the image (ours): each of the 16 path tokens attends over the 49
    image cells (cross-attention), so a stretch of the path can ask what the ink looks like
    around it. Pooled image features, attended path features and the availability flag meet
    in one head."""
    img, tr, av = layers.Input((28, 28, 1), name='img'), layers.Input((64, 5), name='traj'), layers.Input((1,), name='avail')
    fm = _cnn_trunk(img)                                         # 7 x 7 x 64
    cells = layers.Reshape((49, 64))(fm)
    q = layers.Dense(64)(_traj_trunk(tr))                         # 16 x 64
    a = layers.MultiHeadAttention(num_heads=4, key_dim=16)(q, cells)
    a = layers.LayerNormalization()(layers.Add()([q, a]))
    h = layers.Concatenate()([layers.GlobalAveragePooling2D()(fm), layers.GlobalAveragePooling1D()(a), av])
    return keras.Model({'img': img, 'traj': tr, 'avail': av}, _head(h, name), name=name)


JOINT_BUILDERS = {'ordermap_cnn': ordermap_cnn, 'dual_stream': dual_stream, 'ink_glance': ink_glance}
JOINT_INPUTS = {'ordermap_cnn': ('img', 'omap'), 'dual_stream': ('img', 'traj', 'avail'), 'ink_glance': ('img', 'traj', 'avail')}
