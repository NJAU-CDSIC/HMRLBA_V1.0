import os
import argparse
import pandas as pd
import torch
import numpy as np
from utils import logger, get_metrics_reg
from utils.train import train, val
from params import SEED, DEVICE, HP
from graph_loader import CustomTrial, CustomDataLoader, load_data, pad_sequences
from models import DTIProtGraphChemGraphECFP


if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def load_fold_data(dataset, fold_num):
    """Load data for a specific fold using graph_loader's load_data approach"""
    import ast
    import pickle

    # Load and preprocess dataframe
    df_full = pd.read_csv(f'data/{dataset}/updated_full.csv')
    df_full['series'] = df_full['series'].apply(ast.literal_eval)
    maxlen = 1000
    df_full['series'] = list(pad_sequences(df_full['series'].tolist(), maxlen=maxlen))

    # Load fold indices
    with open(f'data/{dataset}/folds/train_fold{fold_num}.txt', 'r') as f:
        train_indices = [int(line.strip()) for line in f]
    with open(f'data/{dataset}/folds/val_fold{fold_num}.txt', 'r') as f:
        val_indices = [int(line.strip()) for line in f]
    with open(f'data/{dataset}/folds/test_fold_all.txt', 'r') as f:
        test_indices = [int(line.strip()) for line in f]

    df_train = df_full.loc[train_indices]
    df_val = df_full.loc[val_indices]
    df_test = df_full.loc[test_indices]

    # Load preprocessed graphs
    with open(f'data/{dataset}/protein_to_graph.pkl', 'rb') as f:
        protein_to_graph = pickle.load(f)
    with open(f'data/{dataset}/ligand_to_graph.pkl', 'rb') as f:
        ligand_to_graph = pickle.load(f)
    with open(f'data/{dataset}/ligand_to_ecfp.pkl', 'rb') as f:
        ligand_to_ecfp = pickle.load(f)

    return df_train, df_val, df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp


def train_single_fold(fold_num, epochs=800, batch_size=128, lr=0.0001, save_dir='results_casf'):
    """Train a single fold"""
    logger.info(f"{'='*80}")
    logger.info(f"Starting Fold {fold_num}/5")
    logger.info(f"{'='*80}")

    # Load data
    dataset = 'casf'
    df_train, df_val, df_test, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_fold_data(dataset, fold_num)

    logger.info(f"Fold {fold_num} - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Create data loaders
    train_dl = CustomDataLoader(df=df_train, batch_size=batch_size, device=DEVICE,
                                e1_key_to_graph=ligand_to_graph,
                                e2_key_to_graph=protein_to_graph,
                                e1_key_to_fp=ligand_to_ecfp,
                                shuffle=True)

    val_dl = CustomDataLoader(df=df_val, batch_size=batch_size, device=DEVICE,
                              e1_key_to_graph=ligand_to_graph,
                              e2_key_to_graph=protein_to_graph,
                              e1_key_to_fp=ligand_to_ecfp,
                              shuffle=False)

    test_dl = CustomDataLoader(df=df_test, batch_size=batch_size, device=DEVICE,
                               e1_key_to_graph=ligand_to_graph,
                               e2_key_to_graph=protein_to_graph,
                               e1_key_to_fp=ligand_to_ecfp,
                               shuffle=False)

    # Initialize model
    logger.info(f"Fold {fold_num} - Initializing model...")
    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop
    logger.info(f"Fold {fold_num} - Starting training for {epochs} epochs...")
    torch.cuda.empty_cache()

    best_val_mse = float('inf')
    best_test_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        y_true_train, y_pred_train, loss_train = train(model, train_dl, optimizer, criterion)

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            y_true_val, y_pred_val, loss_val = val(model, val_dl, criterion)
            y_true_test, y_pred_test, loss_test = val(model, test_dl, criterion)

            metrics_train = get_metrics_reg(y_true_train, y_pred_train)
            metrics_val = get_metrics_reg(y_true_val, y_pred_val)
            metrics_test = get_metrics_reg(y_true_test, y_pred_test)

            current_val_mse = metrics_val['mse']
            if current_val_mse < best_val_mse:
                best_val_mse = current_val_mse
                best_test_metrics = metrics_test
                best_epoch = epoch

                # Save best model for this fold
                os.makedirs(save_dir, exist_ok=True)
                model_path = f'{save_dir}/best_model_fold{fold_num}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_mse': best_val_mse,
                    'metrics_test': metrics_test,
                }, model_path)

            # Log metrics
            logger.info("=" * 80)
            logger.info(f"Fold {fold_num} | Epoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {loss_train:.4f} | RMSE: {metrics_train['rmse']:.4f} | MAE: {metrics_train['mae']:.4f}")
            logger.info(f"Val Loss:   {loss_val:.4f} | RMSE: {metrics_val['rmse']:.4f} | MAE: {metrics_val['mae']:.4f}")
            logger.info(f"Test Loss:  {loss_test:.4f} | RMSE: {metrics_test['rmse']:.4f} | MAE: {metrics_test['mae']:.4f} | "
                       f"Pearson: {metrics_test['pearson']:.4f} | Spearman: {metrics_test['spearman']:.4f} | R2: {metrics_test['r2']:.4f}")
            logger.info(f"Best Val MSE: {best_val_mse:.4f} (Epoch {best_epoch})")

    # Log fold completion
    logger.info("=" * 80)
    logger.info(f"Fold {fold_num} COMPLETED!")
    logger.info(f"Best Epoch: {best_epoch}/{epochs}")
    logger.info(f"Test RMSE: {best_test_metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {best_test_metrics['mae']:.4f}")
    logger.info(f"Test Pearson: {best_test_metrics['pearson']:.4f}")
    logger.info(f"Test Spearman: {best_test_metrics['spearman']:.4f}")
    logger.info(f"Test R²: {best_test_metrics['r2']:.4f}")
    logger.info("=" * 80)

    # Save individual fold result immediately
    fold_result = {
        'Fold': fold_num,
        'Best_Epoch': best_epoch,
        'RMSE': best_test_metrics['rmse'],
        'MAE': best_test_metrics['mae'],
        'Pearson': best_test_metrics['pearson'],
        'Spearman': best_test_metrics['spearman'],
        'R2': best_test_metrics['r2'],
        'MSE': best_test_metrics['mse']
    }

    # Append to results file
    os.makedirs(save_dir, exist_ok=True)
    result_file = f'{save_dir}/fold_results_progress.csv'
    df_result = pd.DataFrame([fold_result])
    if os.path.exists(result_file):
        df_result.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df_result.to_csv(result_file, mode='w', header=True, index=False)
    logger.info(f"Saved Fold {fold_num} results to {result_file}")

    return best_test_metrics, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Train MEGDTA with CASF CV on CASF dataset')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--save_dir', type=str, default='results_casf', help='Directory to save results')
    parser.add_argument('--folds', type=str, default='1,2,3,4,5', help='Comma-separated fold numbers to train')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # Parse folds to train
    folds_to_train = [int(f) for f in args.folds.split(',')]
    logger.info(f"Training folds: {folds_to_train}")

    all_fold_results = []

    for fold_num in folds_to_train:
        best_metrics, best_epoch = train_single_fold(
            fold_num, args.epochs, args.batch_size, args.lr, args.save_dir
        )

        fold_result = {
            'Fold': fold_num,
            'Best_Epoch': best_epoch,
            'RMSE': best_metrics['rmse'],
            'MAE': best_metrics['mae'],
            'Pearson': best_metrics['pearson'],
            'Spearman': best_metrics['spearman'],
            'R2': best_metrics['r2'],
            'MSE': best_metrics['mse']
        }
        all_fold_results.append(fold_result)

    # Save all results
    df_results = pd.DataFrame(all_fold_results)
    os.makedirs(args.save_dir, exist_ok=True)
    df_results.to_csv(f'{args.save_dir}/casf_results.csv', index=False)

    # Calculate average metrics
    logger.info("=" * 80)
    logger.info("CASF CROSS-VALIDATION COMPLETED!")
    logger.info("=" * 80)
    logger.info("Average Test Metrics:")
    logger.info(f"RMSE:     {df_results['RMSE'].mean():.4f} ± {df_results['RMSE'].std():.4f}")
    logger.info(f"MAE:      {df_results['MAE'].mean():.4f} ± {df_results['MAE'].std():.4f}")
    logger.info(f"Pearson:  {df_results['Pearson'].mean():.4f} ± {df_results['Pearson'].std():.4f}")
    logger.info(f"Spearman: {df_results['Spearman'].mean():.4f} ± {df_results['Spearman'].std():.4f}")
    logger.info(f"R²:       {df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
