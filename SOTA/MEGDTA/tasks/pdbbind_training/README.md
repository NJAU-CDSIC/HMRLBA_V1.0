# PDBbind Training

This task trains and evaluates MEGDTA on processed PDBbind-related benchmark
data. Identity splits share one entry point; scaffold and CASF-2016 workflows
are kept as separate scripts.

Run from the MEGDTA SOTA package root:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_pdbbind_identity.py \
  --split identity30 \
  --epochs 800 \
  --fold 0 \
  --save_dir models_identity30 \
  --gpu 0
```

Identity60 split example:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_pdbbind_identity.py \
  --split identity60 \
  --epochs 800 \
  --fold 0 \
  --save_dir models_identity60 \
  --gpu 0
```

Scaffold split example:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_pdbbind_scaffold.py \
  --epochs 800 \
  --gpu 0
```

CASF-2016 workflow:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/prepare_casf_data.py

PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/train_casf.py \
  --epochs 800 \
  --save_dir results_casf \
  --gpu 0
```

To build the extended CASF test set with preprocessed supplement data:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/prepare_casf_data.py \
  --merge_supplement \
  --supplement_pkl supplement_data/preprocessed_supplement.pkl \
  --extended_output_dir data/casf_extended
```

CASF utility commands:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/casf_tools.py check-leakage \
  --data_dir data/casf

PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/casf_tools.py convert-gat \
  --input_dir results_casf \
  --output_dir results_casf_converted

PYTHONPATH=core/megdata:core \
python tasks/pdbbind_training/casf_tools.py evaluate \
  --data_dir data/casf_extended \
  --model_dir results_casf_converted \
  --output_dir results_casf_extended_final
```

Generated checkpoints, logs, prediction CSVs, and metrics files are local
artifacts and should not be committed.
