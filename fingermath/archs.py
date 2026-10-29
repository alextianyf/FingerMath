"""Notebook 12: image-model architectures compared on notebook 09's data (config C).

Every model takes a (28, 28, 1) image, ends in a 64-d 'embedding' layer and a 10-way softmax
'classifier', so they all plug into the same training, evaluation and personalisation code.
"""
import keras
from keras import layers

from . import models


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


def _traj_trunk(t, w=(48, 96)):
    """notebook 11's 1D CNN: returns 16 x w[1] path features."""
    for f in [w[0], w[0]]:
        t = layers.Conv1D(f, 5, padding='same', activation='relu')(t)
    t = layers.MaxPooling1D()(t)
    for f in [w[1], w[1]]:
        t = layers.Conv1D(f, 5, padding='same', activation='relu')(t)
    t = layers.MaxPooling1D()(t)
    return layers.Conv1D(w[1], 3, padding='same', activation='relu')(t)


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


@keras.saving.register_keras_serializable('mathrun')
class CrossAttention(layers.Layer):
    """Multi-head attention written with plain matrix products (MatMul, BatchMatMul, Softmax), the
    same maths as keras' MultiHeadAttention but without Einsum, which TF.js' WASM backend (the
    game's) does not implement. `from_mha` copies the weights of a trained MultiHeadAttention."""

    def __init__(self, num_heads=4, key_dim=16, **kw):
        super().__init__(**kw)
        self.num_heads, self.key_dim = num_heads, key_dim
        inner = num_heads * key_dim
        self.wq, self.wk, self.wv = (layers.Dense(inner, name=n) for n in ('query', 'key', 'value'))
        self.wo = None

    def build(self, q_shape, kv_shape=None):
        self.wo = layers.Dense(q_shape[-1], name='out')
        self.wq.build(q_shape); self.wk.build(kv_shape or q_shape); self.wv.build(kv_shape or q_shape)
        self.wo.build(tuple(q_shape[:-1]) + (self.num_heads * self.key_dim,))

    def _heads(self, x):
        b, n = keras.ops.shape(x)[0], keras.ops.shape(x)[1]
        return keras.ops.transpose(keras.ops.reshape(x, (b, n, self.num_heads, self.key_dim)), (0, 2, 1, 3))

    def call(self, q, kv):
        qh = self._heads(self.wq(q)) * (1.0 / self.key_dim ** 0.5)
        kh, vh = self._heads(self.wk(kv)), self._heads(self.wv(kv))
        att = keras.ops.softmax(keras.ops.matmul(qh, keras.ops.transpose(kh, (0, 1, 3, 2))), axis=-1)
        o = keras.ops.transpose(keras.ops.matmul(att, vh), (0, 2, 1, 3))
        b, n = keras.ops.shape(o)[0], keras.ops.shape(o)[1]
        return self.wo(keras.ops.reshape(o, (b, n, self.num_heads * self.key_dim)))

    def from_mha(self, mha):
        get = lambda l: [w.numpy() for w in l.weights]
        for mine, theirs in [(self.wq, mha._query_dense), (self.wk, mha._key_dense), (self.wv, mha._value_dense)]:
            k, b = get(theirs)                                   # (d, heads, key_dim), (heads, key_dim)
            mine.set_weights([k.reshape(k.shape[0], -1), b.reshape(-1)])
        k, b = get(mha._output_dense)                            # (heads, key_dim, d), (d,)
        self.wo.set_weights([k.reshape(-1, k.shape[-1]), b])

    def get_config(self):
        return {**super().get_config(), 'num_heads': self.num_heads, 'key_dim': self.key_dim}


def ink_glance(name='ink_glance', w=32, tw=(48, 96), attention='mha'):
    """The pen path looks at the image (ours): each of the 16 path tokens attends over the 49
    image cells (cross-attention), so a stretch of the path can ask what the ink looks like
    around it. Pooled image features, attended path features and the availability flag meet
    in one head. `w`, `tw`: image / path widths; attention='matmul' builds the WASM-friendly layer."""
    img, tr, av = layers.Input((28, 28, 1), name='img'), layers.Input((64, 5), name='traj'), layers.Input((1,), name='avail')
    d = 2 * w
    fm = _cnn_trunk(img, w)                                      # 7 x 7 x d
    cells = layers.Reshape((49, d))(fm)
    q = layers.Dense(d)(_traj_trunk(tr, tw))                     # 16 x d
    heads, kd = 4, d // 4
    att = layers.MultiHeadAttention(num_heads=heads, key_dim=kd) if attention == 'mha' else CrossAttention(heads, kd)
    a = att(q, cells)
    a = layers.LayerNormalization()(layers.Add()([q, a]))
    h = layers.Concatenate()([layers.GlobalAveragePooling2D()(fm), layers.GlobalAveragePooling1D()(a), av])
    return keras.Model({'img': img, 'traj': tr, 'avail': av}, _head(h, name), name=name)


def wasm_ink_glance(model, w=32, tw=(48, 96)):
    """Rebuild a trained ink_glance with CrossAttention in place of MultiHeadAttention, same weights."""
    new = ink_glance(model.name, w, tw, attention='matmul')
    for a, b in zip(model.layers, new.layers):
        if isinstance(a, layers.MultiHeadAttention):
            b.from_mha(a)
        else:
            b.set_weights(a.get_weights())
    return new


JOINT_BUILDERS = {'ordermap_cnn': ordermap_cnn, 'dual_stream': dual_stream, 'ink_glance': ink_glance}
JOINT_INPUTS = {'ordermap_cnn': ('img', 'omap'), 'dual_stream': ('img', 'traj', 'avail'), 'ink_glance': ('img', 'traj', 'avail')}
