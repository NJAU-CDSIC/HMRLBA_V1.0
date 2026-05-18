"""
Prepare CASF-2016 benchmark data and optionally build an extended test set.

Base workflow:
  Training split: identity30_remaining.xlsx
  Test split:     CASF-2016_PDB_IDs_only.xlsx

Optional extended workflow:
  Merge supplement_data/preprocessed_supplement.pkl into the fixed CASF test set.
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SEED = 42


def write_indices(path, indices):
    with open(path, "w") as handle:
        for idx in indices:
            handle.write(f"{idx}\n")


def prepare_casf(args):
    np.random.seed(SEED)

    pdb_id_dir = Path(args.pdb_id_dir)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    folds_dir = output_dir / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(exist_ok=True)

    casf_test_ids = pd.read_excel(pdb_id_dir / "CASF-2016_PDB_IDs_only.xlsx")["pdb_id"].tolist()
    train_ids = pd.read_excel(pdb_id_dir / "identity30_remaining.xlsx")["pdb_id"].tolist()

    df_full = pd.read_csv(source_dir / "updated_full.csv")
    df_train_all = df_full[df_full["pdb_id"].isin(train_ids)].copy()
    df_test = df_full[df_full["pdb_id"].isin(casf_test_ids)].copy()

    print(f"Source samples: {len(df_full)}")
    print(f"Training samples found: {len(df_train_all)}")
    print(f"CASF test samples found: {len(df_test)}")

    write_indices(folds_dir / "test_fold_all.txt", df_test.index.tolist())

    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(df_train_all), start=1):
        train_df_indices = df_train_all.iloc[train_indices].index.tolist()
        val_df_indices = df_train_all.iloc[val_indices].index.tolist()
        write_indices(folds_dir / f"train_fold{fold_idx}.txt", train_df_indices)
        write_indices(folds_dir / f"val_fold{fold_idx}.txt", val_df_indices)
        print(f"Fold {fold_idx}: train={len(train_df_indices)}, val={len(val_df_indices)}")

    shutil.copy(source_dir / "updated_full.csv", output_dir / "updated_full.csv")
    for filename in ["protein_to_graph.pkl", "ligand_to_graph.pkl", "ligand_to_ecfp.pkl"]:
        shutil.copy(source_dir / filename, output_dir / filename)

    print(f"Prepared CASF dataset at: {output_dir}")


def merge_pickle_dict(base_path, supplement_dict, output_path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    shutil.copy(base_path, tmp_path)
    with open(tmp_path, "rb") as handle:
        merged = pickle.load(handle)
    merged.update(supplement_dict)
    with open(output_path, "wb") as handle:
        pickle.dump(merged, handle)
    tmp_path.unlink()


def merge_supplement(args):
    casf_dir = Path(args.output_dir)
    extended_dir = Path(args.extended_output_dir)
    supplement_path = Path(args.supplement_pkl)
    extended_dir.mkdir(parents=True, exist_ok=True)
    (extended_dir / "folds").mkdir(exist_ok=True)

    with open(supplement_path, "rb") as handle:
        supplement_package = pickle.load(handle)

    supplement_data = supplement_package["data"]
    supplement_df = pd.DataFrame(supplement_data)
    original_df = pd.read_csv(casf_dir / "updated_full.csv")
    if "series" in original_df.columns and "series" not in supplement_df.columns:
        supplement_df["series"] = [[] for _ in range(len(supplement_df))]

    merged_df = pd.concat([original_df, supplement_df], ignore_index=True)
    merged_df.to_csv(extended_dir / "updated_full.csv", index=False)

    merge_pickle_dict(
        casf_dir / "ligand_to_graph.pkl",
        supplement_package["ligand_to_graph"],
        extended_dir / "ligand_to_graph.pkl",
    )
    merge_pickle_dict(
        casf_dir / "ligand_to_ecfp.pkl",
        supplement_package["ligand_to_ecfp"],
        extended_dir / "ligand_to_ecfp.pkl",
    )
    merge_pickle_dict(
        casf_dir / "protein_to_graph.pkl",
        supplement_package["protein_to_graph"],
        extended_dir / "protein_to_graph.pkl",
    )

    with open(casf_dir / "folds" / "test_fold_all.txt", "r") as handle:
        original_test_indices = [int(line.strip()) for line in handle if line.strip()]
    new_test_indices = original_test_indices + list(range(len(original_df), len(merged_df)))
    write_indices(extended_dir / "folds" / "test_fold_all.txt", new_test_indices)

    for fold in range(1, 6):
        for split in ["train", "val"]:
            shutil.copy(
                casf_dir / "folds" / f"{split}_fold{fold}.txt",
                extended_dir / "folds" / f"{split}_fold{fold}.txt",
            )

    print(f"Merged supplement samples: {len(supplement_df)}")
    print(f"Extended CASF dataset written to: {extended_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CASF and optional extended CASF data")
    parser.add_argument("--pdb_id_dir", default="data/HMRLBA_Datasets/PDB_id_list")
    parser.add_argument("--source_dir", default="data/pdbbind_identity30")
    parser.add_argument("--output_dir", default="data/casf")
    parser.add_argument("--merge_supplement", action="store_true")
    parser.add_argument("--supplement_pkl", default="supplement_data/preprocessed_supplement.pkl")
    parser.add_argument("--extended_output_dir", default="data/casf_extended")
    args = parser.parse_args()

    prepare_casf(args)
    if args.merge_supplement:
        merge_supplement(args)


if __name__ == "__main__":
    main()
