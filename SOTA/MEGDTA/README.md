# MEGDTA SOTA Benchmark Tasks for HMRLBA

This package organizes the MEGDTA benchmark work by task. It is intended to be
placed under the HMRLBA repository, for example:

```text
HMRLBA_V1.0/
└── SOTA/
    └── MEGDTA/
```

Only reusable code and documentation are included. Raw data, processed data,
model checkpoints, logs, predictions, and metric outputs are excluded.

## Structure

```text
.
├── core/
│   ├── megdata/                  # Original MEGDTA model, loader, params, train entry
│   └── utils/                    # Shared MEGDTA training and graph utilities
├── tasks/
│   ├── pdbbind_preprocessing/    # Added preprocessing workflow for HMRLBA/PDBbind
│   ├── pdbbind_training/         # PDBbind identity/scaffold/CASF training
│   └── hxk4_prediction/          # HXK4 preprocessing and prediction scripts
```

## Task 1: PDBbind Preprocessing

MEGDTA originally ships preprocessed data. The scripts in
`tasks/pdbbind_preprocessing/` rebuild MEGDTA inputs from HMRLBA/PDBbind raw
files.

Identity30 example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_preprocessing/preprocess_pdbbind.py \
  --pdbbind_dir data/HMRLBA_Datasets/Raw_data/pdbbind \
  --output_dir data/pdbbind_identity30 \
  --split identity30
```

Identity60 example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_preprocessing/preprocess_pdbbind.py \
  --pdbbind_dir data/HMRLBA_Datasets/Raw_data/pdbbind \
  --output_dir data/pdbbind_identity60 \
  --split identity60
```

## Task 2: PDBbind Training

The scripts in `tasks/pdbbind_training/` train MEGDTA using the processed
PDBbind data.

Identity30 example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_pdbbind_identity.py \
  --split identity30 \
  --epochs 800 \
  --fold 0 \
  --save_dir models_identity30 \
  --gpu 0
```

Identity60 example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_pdbbind_identity.py \
  --split identity60 \
  --epochs 800 \
  --fold 0 \
  --save_dir models_identity60 \
  --gpu 0
```

CASF-2016 CASF example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/prepare_casf_data.py

PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_casf.py \
  --epochs 800 \
  --save_dir results_casf \
  --gpu 0
```

To include supplement data in the extended CASF test set, rerun the preparation
step with `--merge_supplement`.

CASF helper commands are grouped in `tasks/pdbbind_training/casf_tools.py`.

## Task 3: HXK4 Prediction

The scripts in `tasks/hxk4_prediction/` preprocess HXK4 and predict it using
a PDBbind-trained MEGDTA model.

Example:

```bash
cd SOTA/MEGDTA
PYTHONPATH=core/megdata:core \
python tasks/hxk4_prediction/preprocess_hxk4.py \
  --data_dir data/hxk4_raw \
  --output_dir data/hxk4

PYTHONPATH=core/megdata:core \
python tasks/hxk4_prediction/predict_hxk4.py \
  --dataset hxk4 \
  --model models_identity30/best_model_fold0.pth \
  --fold 0 \
  --gpu 0
```

## Do Not Commit Generated Files

The `.gitignore` excludes generated artifacts such as:

- `data/`
- `results/` and `results_*/`
- `models_identity30/`
- `*.pth`, `*.pkl`, `*.log`
- `predictions_*.csv`, `metrics_*.json`
- `__pycache__/`

This keeps the uploaded SOTA package focused on reproducible code instead of
machine-specific outputs.
