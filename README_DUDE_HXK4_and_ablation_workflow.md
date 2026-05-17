# DUD-E HXK4 and ablation workflow

This document describes the DUD-E HXK4 virtual-screening workflow and cascade
ablation scripts added to this repository.

## Contents

- `Datasets/Virtual screening/DUD-E/hxk4/`: DUD-E original files for the HXK4 target.
- `Datasets/Raw_data/dude_hxk4/`: HMRLBA-ready raw dataset for the DUD-E HXK4 workflow.
- `scripts/preprocess/prepare_dude_target.py`: converts one DUD-E target into HMRLBA
  `Raw_data/dude_<target>` format.
- `scripts/preprocess/run_dude_hxk4_preprocess.sh`: regenerates HMRLBA graph caches for HXK4.
- `scripts/eval/run_dude_hxk4_eval.sh`: runs virtual-screening evaluation on HXK4.
- `scripts/train/ablation/run_cascade_ablation.py`: non-invasive PLM-stream cascade ablation script.
- `scripts/train/ablation/run_cnnseq_concat_ablation.py`: CNN-Seq concat ablation script.
- Experimental result files are intentionally not included in this upload package.

## DUD-E HXK4 workflow

From the repository root:

```bash
python scripts/preprocess/prepare_dude_target.py \
  --target-dir "Datasets/Virtual screening/DUD-E/hxk4" \
  --out-root Datasets/Raw_data
```

Then generate graph caches:

```bash
bash scripts/preprocess/run_dude_hxk4_preprocess.sh
```

Finally run HXK4 virtual-screening evaluation:

```bash
bash scripts/eval/run_dude_hxk4_eval.sh
```

The command writes new outputs under `results/` by default when users rerun it,
but this repository package does not include precomputed metrics, predictions,
or result summaries.

Important environment variables:

- `PROT`: HMRLBA repository root.
- `DATA_DIR`: HMRLBA dataset root. Default: `$PROT/Datasets`.
- `EXP_DIR`: experiment/checkpoint root. Default: `$PROT/Experiments`.
- `EXP_NAME`: checkpoint experiment name. Default: `run-20241124_204606-r94ymd7y`.
- `ENV_NAME`: conda environment name. Default: `pyg`.
- `PYTHON_BIN`: Python executable inside the environment. Default: `python`.

The shell launchers assume the project runs in the `pyg` conda environment.
They use the current Python when `conda activate pyg` has already been run;
otherwise they try `conda run --no-capture-output -n pyg python`.

## Cascade ablation workflow

The cascade ablation uses the original HMRLBA dataset/config/checkpoint and
patches the model instance at runtime. It does not modify original model source.

```bash
bash scripts/train/ablation/run_cascade_ablation_eval_hxk4.sh
```

For PDBbind training/evaluation splits, use `run_cascade_ablation.py` directly
and provide the original HMRLBA config via `--config-file`.

## Upload note

Only the HXK4 target data are included in this branch. The full DUD-E collection
is not included to keep the repository branch lightweight and easy to clone.
