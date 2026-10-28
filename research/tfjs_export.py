"""Notebook 12c: Keras model -> TF.js graph model (float16 weights), like export_ship.py.

tensorflowjs 4.22 imports tensorflow_decision_forests, tensorflow_hub and jax at the top of its
converters, but only uses them for decision-forest, TF-Hub and JAX models. They don't install
cleanly next to tensorflow 2.21 on Windows, so empty stand-ins are registered before the import.
"""
import shutil
import sys
import types
from pathlib import Path

for _name in ('tensorflow_decision_forests', 'tensorflow_hub', 'jax', 'jax.experimental', 'jax.experimental.jax2tf'):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules['jax.experimental'].jax2tf = sys.modules['jax.experimental.jax2tf']

from tensorflowjs.converters import tf_saved_model_conversion_v2 as _conv  # noqa: E402


def export(model, out_dir, tmp_dir):
    """Write `model` to `out_dir` (model.json + weight shards); returns the size in KB."""
    out_dir, tmp_dir = Path(out_dir), Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    model.export(str(tmp_dir), verbose=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    _conv.convert_tf_saved_model(str(tmp_dir), str(out_dir), signature_def='serving_default',
                                 saved_model_tags='serve', quantization_dtype_map={'float16': '*'})
    return sum(f.stat().st_size for f in out_dir.iterdir()) // 1024
