"""
Train MEGDTA on PDBbind identity split data.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from graph_loader import CustomDataLoader, CustomTrial, load_data
from models import DTIProtGraphChemGraphECFP
from params import DEVICE, HP, SEED
from utils import get_metrics_reg, logger, save_pkl, train_val


if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def train_pdbbind_identity(split, dataset, epochs=800, fold=0, save_dir=None):
    """Train on a processed PDBbind identity split."""
    save_dir = save_dir or f"models_{split}"

    print("=" * 60)
    print(f"Training MEGDTA on PDBbind ({split} split)")
    print("=" * 60)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    print(f"\n1. Loading {dataset} data...")
    df_train_val, df_test, val_folds, _, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    idx_val = val_folds[fold]
    df_train = df_train_val[~df_train_val.index.isin(idx_val)]

    print(f"   Training samples: {len(df_train)}")
    print(f"   Validation samples: {len(df_train_val[df_train_val.index.isin(idx_val)])}")
    print(f"   Test samples: {len(df_test)}")

    train_dl = CustomDataLoader(
        df=df_train,
        batch_size=128,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=True,
    )
    test_dl = CustomDataLoader(
        df=df_test,
        batch_size=128,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=False,
    )

    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"best_model_fold{fold}.pth")

    epoch_to_metrics, best_y_true, best_y_pred = train_val(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=test_dl,
        epochs=epochs,
        score_fn=get_metrics_reg,
        fold=fold,
        verbose=True,
        with_rm2=True,
        with_ci=True,
        val_nth_epoch=1,
        save_model_path=model_path,
    )

    os.makedirs("results", exist_ok=True)
    save_pkl(epoch_to_metrics, f"results/pdbbind_{split}-fold_{fold}-results.pkl")

    pd.DataFrame({"True_Value": best_y_true, "Predicted_Value": best_y_pred}).to_csv(
        f"fold_{fold}_best_predictions_pdbbind_{split}.csv",
        index=False,
    )

    best_epoch = min(epoch_to_metrics.keys(), key=lambda k: epoch_to_metrics[k]["metrics_val"]["mse"])
    best_metrics = epoch_to_metrics[best_epoch]["metrics_val"]

    config = {
        "dataset": dataset,
        "split": split,
        "fold": fold,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "seed": SEED,
        "hyperparameters": HP,
        "best_metrics": {
            "mse": float(best_metrics["mse"]),
            "rmse": float(best_metrics["rmse"]),
            "pearson": float(best_metrics["pearson"]),
            "spearman": float(best_metrics["spearman"]),
            "rm2": float(best_metrics.get("rm2", 0)),
            "ci": float(best_metrics.get("ci", 0)),
        },
        "model_path": model_path,
    }
    config_path = os.path.join(save_dir, f"config_fold{fold}.json")
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)

    print("=" * 60)
    print("Training complete!")
    print(f"Dataset: {dataset} ({split} split)")
    print(f"Best epoch: {best_epoch}/{epochs}")
    print(f"RMSE: {best_metrics['rmse']:.4f}")
    print(f"Pearson: {best_metrics['pearson']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {config_path}")
    print("=" * 60)

    return model_path, config_path


def main():
    parser = argparse.ArgumentParser(description="Train MEGDTA on PDBbind identity split data")
    parser.add_argument("--split", choices=["identity30", "identity60"], default="identity30")
    parser.add_argument("--dataset", type=str, default=None, help="Processed dataset name under data/")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = args.dataset or f"pdbbind_{args.split}"
    train_pdbbind_identity(args.split, dataset, args.epochs, args.fold, args.save_dir)


if __name__ == "__main__":
    main()
