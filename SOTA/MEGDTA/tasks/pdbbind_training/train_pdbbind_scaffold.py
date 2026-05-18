import os
import argparse
import pandas as pd
import torch
import numpy as np
from utils import logger, get_metrics_reg, save_pkl
from utils.train import train, val
from params import SEED, DEVICE, HP
from graph_loader import CustomTrial, CustomDataLoader, load_data
from models import DTIProtGraphChemGraphECFP


if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def train_single_fold(dataset, epochs=200, batch_size=128, lr=0.0001, save_path=None):
    """Train model on pdbbind dataset with scaffold split"""
    logger.info(f"Loading {dataset} dataset...")
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    # Use the first validation fold
    idx_val = val_folds[0]
    df_train = df_train_val[~df_train_val.index.isin(idx_val)]
    df_val = df_train_val[df_train_val.index.isin(idx_val)]

    logger.info(f"Dataset split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

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

    # Initialize model
    logger.info("Initializing model...")
    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    torch.cuda.empty_cache()

    best_val_mse = float('inf')
    best_y_true_val = None
    best_y_pred_val = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train
        y_true_train, y_pred_train, loss_train = train(model, train_dl, optimizer, criterion)

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            y_true_val, y_pred_val, loss_val = val(model, val_dl, criterion)

            metrics_train = get_metrics_reg(y_true_train, y_pred_train)
            metrics_val = get_metrics_reg(y_true_val, y_pred_val, with_rm2=True, with_ci=True)

            current_val_mse = metrics_val['mse']
            if current_val_mse < best_val_mse:
                best_val_mse = current_val_mse
                best_y_true_val = y_true_val.cpu().numpy().flatten()
                best_y_pred_val = y_pred_val.cpu().numpy().flatten()
                best_epoch = epoch

                # Save best model
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_mse': best_val_mse,
                        'metrics_val': metrics_val,
                    }, save_path)
                    logger.info(f"Saved best model to {save_path}")

            # Log metrics
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {loss_train:.4f}")
            logger.info(f"Train | RMSE: {metrics_train['rmse']:.4f} | MAE: {metrics_train['mae']:.4f} | "
                       f"Pearson: {metrics_train['pearson']:.4f} | Spearman: {metrics_train['spearman']:.4f} | "
                       f"R2: {metrics_train['r2']:.4f}")
            logger.info(f"Val Loss:   {loss_val:.4f}")
            logger.info(f"Val   | RMSE: {metrics_val['rmse']:.4f} | MAE: {metrics_val['mae']:.4f} | "
                       f"Pearson: {metrics_val['pearson']:.4f} | Spearman: {metrics_val['spearman']:.4f} | "
                       f"R2: {metrics_val['r2']:.4f}")
            logger.info(f"Val   | Rm2: {metrics_val['rm2']:.4f} | CI: {metrics_val['ci']:.4f}")
            logger.info(f"Best Val MSE so far: {best_val_mse:.4f} (Epoch {best_epoch})")

    # Save final results
    os.makedirs("results/", exist_ok=True)

    # Save best predictions
    df_pred = pd.DataFrame({
        'True_Value': best_y_true_val,
        'Predicted_Value': best_y_pred_val
    })
    pred_file = f'results/{dataset}_scaffold_best_predictions.csv'
    df_pred.to_csv(pred_file, index=False)
    logger.info(f"Saved best predictions to {pred_file}")

    # Calculate final metrics
    final_metrics = get_metrics_reg(
        torch.tensor(best_y_true_val),
        torch.tensor(best_y_pred_val),
        with_rm2=True,
        with_ci=True
    )

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        'Dataset': dataset,
        'Split': 'scaffold',
        'Epochs': epochs,
        'Best_Epoch': best_epoch,
        'Train_Samples': len(df_train),
        'Val_Samples': len(df_val),
        'RMSE': final_metrics['rmse'],
        'MAE': final_metrics['mae'],
        'Pearson': final_metrics['pearson'],
        'Spearman': final_metrics['spearman'],
        'R2': final_metrics['r2'],
        'MSE': final_metrics['mse'],
        'Rm2': final_metrics['rm2'],
        'CI': final_metrics['ci']
    }])

    summary_file = f'results/{dataset}_scaffold_metrics.csv'
    metrics_summary.to_csv(summary_file, index=False)
    logger.info(f"Saved metrics summary to {summary_file}")

    # Print final summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Epoch: {best_epoch}/{epochs}")
    logger.info(f"RMSE:     {final_metrics['rmse']:.4f}")
    logger.info(f"MAE:      {final_metrics['mae']:.4f}")
    logger.info(f"Pearson:  {final_metrics['pearson']:.4f}")
    logger.info(f"Spearman: {final_metrics['spearman']:.4f}")
    logger.info(f"R²:       {final_metrics['r2']:.4f}")
    logger.info(f"Rm²:      {final_metrics['rm2']:.4f}")
    logger.info(f"CI:       {final_metrics['ci']:.4f}")
    logger.info("=" * 80)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MEGDTA on PDBbind with scaffold split')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--save_path', type=str, default='best_model_pdbbind_scaffold.pth',
                       help='Path to save best model')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_single_fold('pdbbind', args.epochs, args.batch_size, args.lr, args.save_path)


if __name__ == "__main__":
    main()
