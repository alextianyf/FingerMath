# FingerMath

Handwritten math answers, recognised on the device: written with a finger or a pen, read in the browser, and checked against the expected answer.

This repository holds the recognition research behind [Math Run](https://github.com/alextianyf/Math-Run), a 3D browser game where you answer arithmetic by writing the digits. It started as `mlp_training/` in that repository and moved here with its history. The game keeps only the shipped models (`digit_model/`, `mnist_tfjs_model/`).

So far the work covers single digits (0–9). Next come multi-digit answers and one-line arithmetic (`fingermath/multidigit.py` and `scripts/compare_multi.py` are a parked prototype).

## Layout

```
FingerMath/
├── fingermath/        # The Python package: data, rendering, models, training recipes
├── scripts/           # Command-line entry points: train, export to TF.js, build data
├── notebooks/         # Research notebooks 01–14
├── reports/           # Notes, literature review and the research report
├── results/           # JSON results written by the notebooks and scripts
├── exports/           # TF.js exports and the check files for the browser benchmark
├── web/               # Browser pages: sample collection, end-to-end check, benchmarks
├── baselines/mlp/     # The original MNIST MLP: notebook, logs, SavedModel
├── data/MNIST.zip     # MNIST (Yann LeCun et al., CC BY-SA 3.0)
└── cache/             # Downloaded datasets and trained models (git-ignored, ~600 MB)
```

## Notebooks

| # | Topic |
|---|---|
| 01 | Domain shift: pad handwriting is not MNIST |
| 02 | Baseline: the deployed MLP under domain shift |
| 03 | A small CNN, augmentation and pad-style training data |
| 04 | Personalised few-shot fine-tuning |
| 07 | Real handwriting benchmarks |
| 08 | Existing models on real handwriting |
| 09 | Training with real data and targeted augmentation |
| 10 | Error audit |
| 11 | A pen-trajectory model, fused with the image model |
| 12, 12b, 12c | Image architectures; one model for image and ink; distillation and browser cost |
| 13 | Confidence calibration and the game's decision rule |
| 14 | Personalisation inside the game |

Reports: [06](reports/06_recognition.md) the shipped CNN, [07](reports/07_literature_and_proposal.md) literature review and proposal, [15](reports/15_report.md) summary of 07–14 and the recommendation for the game. Most notes from 06 on are written in Chinese.

## Setup

Python 3.13:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install --no-deps tensorflowjs==4.22.0 tf_keras==2.21.0
.venv/Scripts/python -m pip install -e .
```

Notebooks run from `notebooks/` and scripts from the repository root, for example `python scripts/train_ship.py final`. `baselines/mlp/requirements.txt` is the older environment of the original MLP notebook.

## Working with the game

Some tools use the game's own code and models, so check out Math Run next to this repository:

```
Projects/
├── FingerMath/
└── Math-Run/
```

- The pages in `web/` import `Math-Run/js/` and its models. Serve the parent folder (`Projects/`) with any static server, for example `python -m http.server`, and open `/FingerMath/web/collect/`.
- `scripts/export_ship.py` writes the shipped model into `Math-Run/digit_model/`. Set `MATH_RUN` to use another path.

## License

The code is released under the [MIT License](LICENSE). Third-party datasets keep their own licenses: MNIST is CC BY-SA 3.0, and the real-handwriting datasets that notebook 07 downloads into `cache/` are not redistributed here.
