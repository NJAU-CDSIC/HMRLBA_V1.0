#!/usr/bin/env python
"""
Utility commands for CASF/PDBbind benchmark maintenance.

Subcommands:
  check-leakage       Check CASF split overlap.
  convert-gat         Convert old PyG GATConv lin_src/lin_dst weights.
  count-parameters    Count parameters in a checkpoint.
  evaluate            Evaluate 5 fold checkpoints on a CASF-style test set.
"""

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from graph_loader import CustomDataLoader, CustomTrial, pad_sequences
from models import DTIProtGraphChemGraphECFP
from params import HP


def load_indices(path):
    with open(path, "r") as handle:
        return set(int(line.strip()) for line in handle if line.strip())


def check_leakage(args):
    data_dir = Path(args.data_dir)
    folds_dir = data_dir / "folds"
    df = pd.read_csv(data_dir / "updated_full.csv")
    test_indices = load_indices(folds_dir / "test_fold_all.txt")

    all_train_indices = set()
    all_val_indices = set()
    has_leakage = False

    print(f"Total samples: {len(df)}")
    print(f"Test samples: {len(test_indices)}")

    for fold in range(1, 6):
        train_indices = load_indices(folds_dir / f"train_fold{fold}.txt")
        val_indices = load_indices(folds_dir / f"val_fold{fold}.txt")

        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices

        print(f"\nFold {fold}")
        print(f"  Train: {len(train_indices)}")
        print(f"  Val:   {len(val_indices)}")
        print(f"  Train/val overlap:  {len(train_val_overlap)}")
        print(f"  Train/test overlap: {len(train_test_overlap)}")
        print(f"  Val/test overlap:   {len(val_test_overlap)}")

        has_leakage = has_leakage or bool(train_val_overlap or train_test_overlap or val_test_overlap)
        all_train_indices.update(train_indices)
        all_val_indices.update(val_indices)

    train_val_test_overlap = (all_train_indices | all_val_indices) & test_indices
    missing = set(range(len(df))) - (all_train_indices | all_val_indices | test_indices)
    has_leakage = has_leakage or bool(train_val_test_overlap)

    print("\nSummary")
    print(f"  Cross-fold train/val vs test overlap: {len(train_val_test_overlap)}")
    print(f"  Unused samples: {len(missing)}")
    print("  Status: " + ("LEAKAGE FOUND" if has_leakage else "OK"))


def convert_gat_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lin_src.weight" in key:
            lin_dst_key = key.replace("lin_src.weight", "lin_dst.weight")
            if lin_dst_key in state_dict:
                new_state_dict[key.replace("lin_src.weight", "lin.weight")] = value
        elif "lin_dst.weight" not in key:
            new_state_dict[key] = value
    return new_state_dict


def convert_gat(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        model_path = input_dir / f"best_model_fold{fold}.pth"
        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint["model_state_dict"] = convert_gat_state_dict(checkpoint["model_state_dict"])
        output_path = output_dir / f"best_model_fold{fold}.pth"
        torch.save(checkpoint, output_path)
        print(f"Converted fold {fold}: {output_path}")


def count_parameters(args):
    checkpoint = torch.load(args.model, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    total_params = sum(param.numel() for param in state_dict.values())

    print(f"Model file: {args.model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameters (M): {total_params / 1e6:.2f}M")

    module_params = {}
    for name, param in state_dict.items():
        module_name = name.split(".")[0]
        module_params[module_name] = module_params.get(module_name, 0) + param.numel()

    print("\nParameter breakdown by module:")
    for module_name, num_params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        print(f"{module_name:30s}: {num_params:12,} ({num_params / 1e6:6.2f}M)")


def load_test_data(data_dir):
    data_dir = Path(data_dir)
    df_full = pd.read_csv(data_dir / "updated_full.csv")
    df_full["series"] = df_full["series"].apply(ast.literal_eval)
    df_full["series"] = list(pad_sequences(df_full["series"].tolist(), maxlen=1000))

    with open(data_dir / "folds" / "test_fold_all.txt", "r") as handle:
        test_indices = [int(line.strip()) for line in handle]
    df_test = df_full.loc[test_indices]

    with open(data_dir / "protein_to_graph.pkl", "rb") as handle:
        protein_to_graph = pickle.load(handle)
    with open(data_dir / "ligand_to_graph.pkl", "rb") as handle:
        ligand_to_graph = pickle.load(handle)
    with open(data_dir / "ligand_to_ecfp.pkl", "rb") as handle:
        ligand_to_ecfp = pickle.load(handle)

    return df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp


def evaluate_one_model(model_path, df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp, device):
    test_loader = CustomDataLoader(
        df=df_test,
        batch_size=128,
        device=device,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=False,
    )

    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    predictions, labels = [], []
    with torch.no_grad():
        for y_true, data_batch in test_loader:
            y_pred = model(data_batch)
            predictions.extend(y_pred.cpu().numpy())
            labels.extend(y_true.cpu().numpy())

    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "rmse": np.sqrt(mean_squared_error(labels, predictions)),
        "mae": mean_absolute_error(labels, predictions),
        "pearson": pearsonr(labels, predictions)[0],
        "spearman": spearmanr(labels, predictions)[0],
        "r2": r2_score(labels, predictions),
        "predictions": predictions,
        "labels": labels,
    }


def evaluate(args):
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_test_data(data_dir)

    all_results = []
    all_predictions = {}

    for fold in range(1, 6):
        model_path = model_dir / f"best_model_fold{fold}.pth"
        result = evaluate_one_model(model_path, df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp, device)

        print(
            f"Fold {fold}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
            f"Pearson={result['pearson']:.4f}, Spearman={result['spearman']:.4f}, R2={result['r2']:.4f}"
        )

        all_results.append({key: result[key] for key in ["rmse", "mae", "pearson", "spearman", "r2"]} | {"fold": fold})
        all_predictions[f"fold{fold}"] = result["predictions"]
        pd.DataFrame({"label": result["labels"], "prediction": result["predictions"]}).to_csv(
            output_dir / f"fold{fold}_predictions.csv",
            index=False,
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "cross_validation_summary.csv", index=False)

    pd.DataFrame({"label": result["labels"], **all_predictions}).to_csv(
        output_dir / "all_folds_predictions.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(description="CASF/PDBbind benchmark helper tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    leakage_parser = subparsers.add_parser("check-leakage")
    leakage_parser.add_argument("--data_dir", default="data/casf")
    leakage_parser.set_defaults(func=check_leakage)

    convert_parser = subparsers.add_parser("convert-gat")
    convert_parser.add_argument("--input_dir", default="results_casf")
    convert_parser.add_argument("--output_dir", default="results_casf_converted")
    convert_parser.set_defaults(func=convert_gat)

    count_parser = subparsers.add_parser("count-parameters")
    count_parser.add_argument("--model", required=True)
    count_parser.set_defaults(func=count_parameters)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--data_dir", default="data/casf_extended")
    eval_parser.add_argument("--model_dir", default="results_casf_converted")
    eval_parser.add_argument("--output_dir", default="results_casf_extended_final")
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
