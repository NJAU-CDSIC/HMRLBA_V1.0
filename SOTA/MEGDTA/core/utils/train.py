import torch
from .graph_batcher import logger


def train(model, data_loader, optimizer, criterion):
    y_true, y_pred, losses = [], [], []
    model.train()
    for y_train, data_batch in data_loader:
        optimizer.zero_grad()
        y_train_pred = model(data_batch)

        train_loss = criterion(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        losses.append(train_loss.item())
        y_true.append(y_train)
        y_pred.append(y_train_pred)

    epoch_loss = sum(losses) / len(losses)
    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()

    return y_true, y_pred, epoch_loss


def val(model, data_loader, criterion):
    y_true, y_pred, losses = [], [], []
    model.eval()
    for y_val, data_batch in data_loader:
        with torch.no_grad():
            y_val_pred = model(data_batch)
            val_loss = criterion(y_val_pred, y_val)
        losses.append(val_loss.item())
        y_true.append(y_val)
        y_pred.append(y_val_pred)

    epoch_loss = sum(losses) / len(losses)
    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()
    return y_true, y_pred, epoch_loss


def train_val(model, optimizer, criterion, train_dl, val_dl, epochs, score_fn, fold=None, verbose=False,
              with_rm2=False, with_ci=False, val_nth_epoch=10, save_model_path=None):
    torch.cuda.empty_cache()

    epoch_to_metrics = {}
    best_val_mse = float('inf')
    best_y_true_val = None
    best_y_pred_val = None
    best_model_state = None

    for epoch in range(epochs):
        epoch += 1

        y_true_train, y_pred_train, loss_train = train(model, train_dl, optimizer, criterion)

        if epoch % val_nth_epoch != 0:
            continue

        y_true_val, y_pred_val, loss_val = val(model, val_dl, criterion)

        metrics_train = score_fn(y_true_train, y_pred_train)
        metrics_val = score_fn(y_true_val, y_pred_val, with_rm2=with_rm2, with_ci=with_ci)

        current_val_mse = metrics_val['mse']
        if current_val_mse < best_val_mse:
            best_val_mse = current_val_mse
            best_y_true_val = y_true_val.cpu().numpy().flatten() if isinstance(y_true_val,
                                                                               torch.Tensor) else y_true_val.flatten()
            best_y_pred_val = y_pred_val.cpu().numpy().flatten() if isinstance(y_pred_val,
                                                                               torch.Tensor) else y_pred_val.flatten()
            best_model_state = model.state_dict().copy()

        epoch_to_metrics[epoch] = {}
        epoch_to_metrics[epoch]["metrics_train"] = metrics_train
        epoch_to_metrics[epoch]["metrics_val"] = metrics_val
        epoch_to_metrics[epoch]["loss_train"] = loss_train
        epoch_to_metrics[epoch]["loss_val"] = loss_val

        if verbose:
            if fold is not None:
                logger.info(f"=" * 80)
                logger.info(f"Fold {fold} | Epoch {epoch}/{epochs}")
            else:
                logger.info(f"=" * 80)
                logger.info(f"Epoch {epoch}/{epochs}")

            # Train metrics
            train_metrics = epoch_to_metrics[epoch]["metrics_train"]
            logger.info(f"Train Loss: {epoch_to_metrics[epoch]['loss_train']:.4f}")
            logger.info(f"Train | RMSE: {train_metrics.get('rmse', 0):.4f} | MAE: {train_metrics.get('mae', 0):.4f} | "
                       f"Pearson: {train_metrics.get('pearson', 0):.4f} | Spearman: {train_metrics.get('spearman', 0):.4f} | "
                       f"R2: {train_metrics.get('r2', 0):.4f}")

            # Validation metrics
            val_metrics = epoch_to_metrics[epoch]["metrics_val"]
            logger.info(f"Val Loss:   {epoch_to_metrics[epoch]['loss_val']:.4f}")
            logger.info(f"Val   | RMSE: {val_metrics.get('rmse', 0):.4f} | MAE: {val_metrics.get('mae', 0):.4f} | "
                       f"Pearson: {val_metrics.get('pearson', 0):.4f} | Spearman: {val_metrics.get('spearman', 0):.4f} | "
                       f"R2: {val_metrics.get('r2', 0):.4f}")

            if with_rm2 and 'rm2' in val_metrics:
                logger.info(f"Val   | Rm2: {val_metrics['rm2']:.4f}")
            if with_ci and 'ci' in val_metrics:
                logger.info(f"Val   | CI: {val_metrics['ci']:.4f}")

            logger.info(f"Best Val MSE so far: {best_val_mse:.4f}")

    if save_model_path and best_model_state is not None:
        torch.save(best_model_state, save_model_path)
        logger.info(f"Best model saved to {save_model_path}")

    return epoch_to_metrics, best_y_true_val, best_y_pred_val


def train_final(model, optimizer, criterion, train_dl, epochs, score_fn, verbose=True):
    torch.cuda.empty_cache()

    epoch_to_metrics = {}

    for epoch in range(epochs):
        epoch += 1

        y_true_train, y_pred_train, loss_train = train(model, train_dl, optimizer, criterion)
        metrics_train = score_fn(y_true_train, y_pred_train)

        epoch_to_metrics[epoch] = {}
        epoch_to_metrics[epoch]["metrics_train"] = metrics_train
        epoch_to_metrics[epoch]["loss_train"] = loss_train

        if epoch == epochs:
            if verbose:
                logger.info("Train      | " +
                            str({k: round(v, 3) for k, v in epoch_to_metrics[epoch]["metrics_train"].items()}) +
                            " | loss: " + str(round(epoch_to_metrics[epoch]["loss_train"], 3)))
    return epoch_to_metrics