"""
Predict on HXK4 virtual screening dataset and calculate classification metrics.

This script loads a pre-trained model and evaluates it on the HXK4 dataset,
calculating AUROC, AUPR, and enrichment factors.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

from graph_loader import CustomDataLoader, load_data
from models import DTIProtGraphChemGraphECFP
from params import SEED, DEVICE, HP
from utils import logger


def calculate_enrichment_factor(y_true, y_scores, percentage):
    """
    Calculate enrichment factor at a given percentage.

    EF = (Actives_found_at_x% / Total_actives) / (x% / 100%)
    """
    n = len(y_true)
    n_actives = np.sum(y_true)

    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]

    # Get top x%
    top_n = int(n * percentage / 100)
    top_indices = sorted_indices[:top_n]

    # Count actives in top x%
    actives_found = np.sum(y_true[top_indices])

    # Calculate EF
    ef = (actives_found / n_actives) / (percentage / 100)

    return ef


def predict_and_evaluate(dataset, model_path=None, fold=0):
    """
    Run prediction on HXK4 dataset and calculate metrics.

    Args:
        dataset: Dataset name (e.g., 'hxk4')
        model_path: Path to pre-trained model (optional)
        fold: Fold number (default: 0)
    """

    print("=" * 60)
    print(f"Predicting on {dataset} dataset")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    print(f"   Test set size: {len(df_test)}")
    print(f"   Actives: {(df_test['label'] == 1).sum()}")
    print(f"   Decoys: {(df_test['label'] == 0).sum()}")

    # Create data loader
    test_dl = CustomDataLoader(
        df=df_test,
        batch_size=128,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=False
    )

    # Initialize model
    print("\n2. Initializing model...")
    from graph_loader import CustomTrial
    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)

    # Load pre-trained weights if provided
    if model_path and os.path.exists(model_path):
        print(f"   Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("   Using randomly initialized model (no pre-trained weights)")
        print("   Note: For meaningful predictions, you should train the model first")

    # Run prediction
    print("\n3. Running predictions...")
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="   Predicting"):
            labels, features = batch

            # Forward pass
            predictions = model(features)

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    print(f"   Predictions shape: {all_predictions.shape}")
    print(f"   Labels shape: {all_labels.shape}")

    # Calculate metrics
    print("\n4. Calculating metrics...")

    # AUROC
    auroc = roc_auc_score(all_labels, all_predictions)
    print(f"   AUROC: {auroc:.4f}")

    # AUPR (Average Precision)
    aupr = average_precision_score(all_labels, all_predictions)
    print(f"   AUPR: {aupr:.4f}")

    # Enrichment Factors
    ef1 = calculate_enrichment_factor(all_labels, all_predictions, 1.0)
    ef5 = calculate_enrichment_factor(all_labels, all_predictions, 5.0)
    ef10 = calculate_enrichment_factor(all_labels, all_predictions, 10.0)

    print(f"   EF 1%: {ef1:.4f}")
    print(f"   EF 5%: {ef5:.4f}")
    print(f"   EF 10%: {ef10:.4f}")

    # Save results
    print("\n5. Saving results...")

    # Save predictions
    results_df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_predictions,
        'ligand': df_test['ligand'].values,
        'protein': df_test['protein'].values
    })

    output_file = f'predictions_{dataset}_fold{fold}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"   Saved predictions to: {output_file}")

    # Save metrics
    metrics = {
        'dataset': dataset,
        'fold': fold,
        'n_samples': len(all_labels),
        'n_actives': int(np.sum(all_labels)),
        'n_decoys': int(len(all_labels) - np.sum(all_labels)),
        'auroc': float(auroc),
        'aupr': float(aupr),
        'ef1': float(ef1),
        'ef5': float(ef5),
        'ef10': float(ef10)
    }

    metrics_file = f'metrics_{dataset}_fold{fold}.json'
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to: {metrics_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Actives: {int(np.sum(all_labels))} ({100*np.mean(all_labels):.2f}%)")
    print(f"Decoys: {int(len(all_labels) - np.sum(all_labels))} ({100*(1-np.mean(all_labels)):.2f}%)")
    print()
    print(f"AUROC:  {auroc:.4f}")
    print(f"AUPR:   {aupr:.4f}")
    print(f"EF 1%:  {ef1:.4f}")
    print(f"EF 5%:  {ef5:.4f}")
    print(f"EF 10%: {ef10:.4f}")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Predict on HXK4 virtual screening dataset')
    parser.add_argument('--dataset', type=str, default='hxk4',
                       help='Dataset name (default: hxk4)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model (optional)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number (default: 0)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU id to use (e.g., 0, 1, 2, 3)')

    args = parser.parse_args()

    # Set GPU
    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    # Set random seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # Run prediction
    predict_and_evaluate(args.dataset, args.model, args.fold)


if __name__ == "__main__":
    main()
