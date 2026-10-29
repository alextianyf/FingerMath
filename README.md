# FingerMath

Handwritten math answers, recognised on the device: written with a finger or a pen, read in the browser, and checked against the expected answer.

This repository holds the recognition research behind [Math Run](https://github.com/alextianyf/Math-Run), a 3D browser game where you answer arithmetic by writing the digits. It started as `mlp_training/` in that repository and moved here with its history. The game keeps only the shipped models (`digit_model/`, `mnist_tfjs_model/`).

So far the work covers single digits (0–9). Next come multi-digit answers and one-line arithmetic (see `research/multidigit.py`, a parked prototype).

## Layout

```
FingerMath/
├── research/               # Notebooks 01–14, notes and reports, Python modules
│   ├── 15_report.md        # Summary of 07–14 and the recommendation for the game
│   ├── collect/            # Page for collecting real handwriting samples
│   ├── e2e/                # End-to-end check of the game's recognizer
│   ├── export/             # TF.js exports and the browser benchmark pages
│   ├── results/            # JSON results of each notebook
│   └── cache/              # Datasets and trained models (git-ignored, ~600 MB)
├── mlp_baseline/           # The original MNIST MLP: notebook, logs, SavedModel
└── data/MNIST.zip          # MNIST, used by the MLP notebook and research/datasets.py
```

## Notebooks

| # | Topic |
|---|---|
| 01 | Domain shift: pad handwriting is not MNIST |
| 02 | Baseline: the deployed MLP under domain shift |
| 03 | A small CNN, augmentation and pad-style training data |
| 04 | Personalised few-shot fine-tuning |
| 06 | The shipped CNN (notes) |
| 07 | Real handwriting benchmarks; literature review and proposal |
| 08 | Existing models on real handwriting |
| 09 | Training with real data and targeted augmentation |
| 10 | Error audit |
| 11 | A pen-trajectory model, fused with the image model |
| 12, 12b, 12c | Image architectures; one model for image and ink; distillation and browser cost |
| 13 | Confidence calibration and the game's decision rule |
| 14 | Personalisation inside the game |
| 15 | Report and recommendation |

Most notes from 06 on are written in Chinese.

## Working with the game

Some tools use the game's own code and models, so check out Math Run next to this repository:

```
Projects/
├── FingerMath/
└── Math-Run/
```

- `research/collect/`, `research/e2e/` and `research/export/bench*.html` import `Math-Run/js/` and its models. Serve the parent folder (`Projects/`) with any static server, for example `python -m http.server`, and open `/FingerMath/research/collect/`.
- `research/export_ship.py` writes the shipped model into `Math-Run/digit_model/`. Set `MATH_RUN` to use another path.

## Setup

Python 3.13:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r research/requirements-research.txt
.venv/Scripts/python -m pip install --no-deps tensorflowjs==4.22.0 tf_keras==2.21.0
```

`mlp_baseline/requirements.txt` is the older environment of the original MLP notebook.
