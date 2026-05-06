#attention : c'est pas bon (à modifier)
# train_ds et val_ds j'ai pas fait, jsp ce que c'est 

import numpy as np
import pandas as pd
import torch

import config
from data_loader import (charger_transferts, charger_stats_joueurs) 
from feature_engineering import (
    calculer_medianes_par_poste,
    calculer_score_cible,
    construire_index_joueur,
    construire_toutes_sequences,
    construire_contexte,
    split_temporel,
    standardiser
)

from model import TransferLSTM
from dataset import TransferDataset
from train import entrainer, evaluer, tracer_historique
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ----- 1. CHARGEMENT -----
    print("ÉTAPE 1 — Chargement des données")
    df_t = charger_transferts()
    df_s = charger_stats_joueurs()

    # ----- 2. CONSTRUCTION DU DATASET -----
    print("ÉTAPE 2 — Construction des features et de la cible")
    index = construire_index_joueur(df_s)
    medianes = calculer_medianes_par_poste(df_s)
    X_seq, masques, df_t = construire_toutes_sequences(df_t,index)
    X_ctx = construire_contexte(df_t)
    y = calculer_score_cible(df_t, df_s, medianes)

    print(f"\nDimensions :")
    print(f"  X_seq : {X_seq.shape}  (transferts, saisons, features)")
    print(f"  y     : {y.shape}     min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")



    # ----- 3. SPLIT TEMPOREL -----
    print("ÉTAPE 3 — Split temporel")
    splits = split_temporel(df_t, X_seq, X_ctx, y,masques)

    X_seq_tr, X_ctx_tr, y_tr = splits["train"]
    X_seq_va, X_ctx_va, y_va = splits["val"]
    X_seq_te, X_ctx_te, y_te = splits["test"]

    if len(y_tr) == 0 or len(y_va) == 0 or len(y_te) == 0:
        raise RuntimeError(
            "Un des splits est vide. Vérifier les bornes de saisons dans config.py "
            "ou la couverture du dataset."
        )

    
    # ----- 4. STANDARDISATION -----
    print("ÉTAPE 4 — Standardisation (fit sur train uniquement)")

    # Pour les SÉQUENCES, on standardise les FEATURES_STATS + FEATURES_C + age
    # Tout sauf les one-hot postes et le masque
    n_features_seq_a_norm = len(config.FEATURES_STATS) + len(config.FEATURES_C) + 1

    X_seq_tr, X_seq_va, X_seq_te = standardiser(
       X_seq_tr.copy(), X_seq_va.copy(), X_seq_te.copy(), n_features_seq_a_norm
    )

    # Pour le CONTEXTE, on standardise transfer_fee et market_val (les 2 premières)
    # La 3e (transfert_hiver) est binaire, on ne la standardise pas
    n_features_ctx_a_norm = 2

    X_ctx_tr, X_ctx_va, X_ctx_te = standardiser(
        X_ctx_tr.copy(), X_ctx_va.copy(), X_ctx_te.copy(), n_features_ctx_a_norm
    )

    print("  Standardisation OK")

    # ----- 5. DATASETS PYTORCH -----
    train_ds = TransferDataset(X_seq_tr, X_ctx_tr, y_tr)
    val_ds = TransferDataset(X_seq_va, X_ctx_va, y_va)
    test_ds = TransferDataset(X_seq_te, X_ctx_te, y_te)


    # ----- 6. MODÈLE -----
    print("ÉTAPE 5 — Construction du modèle")
    model = TransferLSTM(
        n_features_seq=X_seq_tr.shape[2],
        n_features_ctx=X_ctx_tr.shape[1],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : {model}")
    print(f"  Nb paramètres : {n_params:,}")

    # ----- 7. ENTRAÎNEMENT -----
    print("ÉTAPE 6 — Entraînement")
    model, historique = entrainer(model, train_ds, val_ds, device)

    # Sauvegarde du modèle
    chemin_modele = config.MODEL_DIR / "lstm_transfert.pt"
    torch.save(model.state_dict(), chemin_modele)
    print(f"\n  Modèle sauvegardé : {chemin_modele}")

    # Courbes d'apprentissage
    tracer_historique(historique, config.OUTPUT_DIR / "courbes_entrainement.png")

    # ----- 8. ÉVALUATION TEST -----
    print("ÉTAPE 7 — Évaluation finale sur le test set")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loss, test_mae, test_rmse, test_r2, y_true, y_pred = evaluer(model, test_loader, device)
    print(f"  Test Loss : {test_loss:.4f}")
    print(f"  Test MAE  : {test_mae:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")
    print(f"  Test R²   : {test_r2:.3f}")

    '''
    # Baseline naïve (prédire la moyenne du train) pour comparaison
    y_naif = np.full_like(y_true, fill_value=y_tr.mean())
    mae_naif = np.mean(np.abs(y_true - y_naif))
    print(f"\n  Baseline (prédire la moyenne) : MAE={mae_naif:.4f}")
    if mae_naif > 0:
        gain = (mae_naif - test_mae) / mae_naif * 100
        print(f"  Gain du LSTM vs baseline : {gain:+.1f}%")
    '''

    # ----- 9. EXPORT DES PRÉDICTIONS -----
    saisons_test = df_t.iloc[
        (df_t["season"] > config.SAISON_VAL_MAX)
        & (df_t["season"] <= config.SAISON_TEST_MAX)
    ].copy() if False else df_t[
        (df_t["season"] > config.SAISON_VAL_MAX)
        & (df_t["season"] <= config.SAISON_TEST_MAX)
    ].copy()

    df_pred = pd.DataFrame({
        "player": saisons_test["player"].values,
        "season": saisons_test["season"].values,
        "from": saisons_test["from"].values,
        "to": saisons_test["to"].values,
        "score_reel": y_true,
        "score_predit": y_pred,
        "ecart": y_pred - y_true,
    })
    chemin_pred = config.OUTPUT_DIR / "predictions_test.csv"
    df_pred.to_csv(chemin_pred, index=False)
    print(f"\n  Prédictions test exportées : {chemin_pred}")


    print("TERMINÉ")



if __name__ == "__main__":
    main()
