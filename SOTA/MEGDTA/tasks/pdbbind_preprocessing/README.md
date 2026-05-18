# PDBbind Preprocessing

This task adds the missing preprocessing workflow for MEGDTA. It converts
HMRLBA/PDBbind raw structural data into MEGDTA-compatible files.

Run from the MEGDTA SOTA package root:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_preprocessing/preprocess_pdbbind.py \
  --pdbbind_dir data/HMRLBA_Datasets/Raw_data/pdbbind \
  --output_dir data/pdbbind_identity30 \
  --split identity30
```

Identity60 example:

```bash
PYTHONPATH=core/megdata:core \
python tasks/pdbbind_preprocessing/preprocess_pdbbind.py \
  --pdbbind_dir data/HMRLBA_Datasets/Raw_data/pdbbind \
  --output_dir data/pdbbind_identity60 \
  --split identity60
```

Generated files under `data/` are local artifacts and should not be committed.
