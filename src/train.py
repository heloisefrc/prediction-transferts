"""
Boucle d'entraînement avec early stopping et logging.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def evaluer(model, loader, device):
    """Évalue le modèle : retourne (loss, MAE, RMSE, R²)."""
    model.eval()
    criterion = nn.MSELoss()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for x_seq, x_ctx, y in loader:
            x_seq, x_ctx, y = x_seq.to(device), x_ctx.to(device), y.to(device)
            out = model(x_seq, x_ctx)
            loss = criterion(out, y)
            losses.append(loss.item())
            y_true.append(y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    return np.mean(losses), mae, rmse, r2, y_true, y_pred


def entrainer(model, train_ds, val_ds, device):
    """Boucle d'entraînement principale avec early stopping."""
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    historique = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.N_EPOCHS + 1):
        model.train()
        train_losses = []
        for x_seq, x_ctx, y in train_loader:
            x_seq, x_ctx, y = x_seq.to(device), x_ctx.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_seq, x_ctx)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, val_mae, val_rmse, val_r2, _, _ = evaluer(model, val_loader, device)
        scheduler.step(val_loss)

        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_mae"].append(val_mae)
        historique["val_rmse"].append(val_rmse)
        historique["val_r2"].append(val_r2)

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | R²={val_r2:.3f}"
        )

        # Early stopping
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping à l'epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, historique


def tracer_historique(historique, chemin_sortie):
    """Sauvegarde les courbes d'apprentissage."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(historique["train_loss"], label="train")
    axes[0].plot(historique["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(historique["val_mae"], label="MAE")
    axes[1].plot(historique["val_rmse"], label="RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Erreur")
    axes[1].set_title("Métriques validation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=120)
    plt.close()
    print(f"  Courbes sauvegardées : {chemin_sortie}")