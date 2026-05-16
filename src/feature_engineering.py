"""
Construction des features et de la cible pour le LSTM.
"""
import pandas as pd
import numpy as np
import config #config.py
import torch

#construction cible

def calculer_medianes_par_poste(df_stats):
    """
    Calcule (G+A)/90 médian pour chaque poste, sur tout le dataset stats.
    Retourne un dict : {'FW': 0.42, 'MF': 0.18, ...}
    """
    df = df_stats.copy()
    
    # On veut un échantillon de joueurs qui jouent vraiment (au moins ~3 matchs titulaires)
    df = df[df["Min"] > 270]
    
    # Calculer (G+A) par 90 minutes
    df["ga_per_90"] = (df["Gls"]+df["Ast"]) / (df["Min"]/90)
    
    # Médiane par poste
    medianes = df.groupby("Pos_main")["ga_per_90"].median().to_dict()
    return medianes


def score_saison(ligne, medianes):
    mp = ligne["MP"]
    minutes = ligne["Min"]

    if not mp or not minutes or pd.isna(mp) or pd.isna(minutes):
        return 0.0

    poste = ligne.get("Pos_main", "")
    poids = config.POIDS_PAR_POSTE.get(poste, config.POIDS_DEFAUT)

    # 1. Temps de jeu
    temps_jeu = min(minutes / mp / 90.0, 1.0)

    # 2. Impact offensif normalisé par poste
    nineties = minutes / 90
    if nineties > 0:
        ga_per_90 = (ligne["Gls"] + ligne["Ast"]) / nineties
    else:
        ga_per_90 = 0.0

    mediane_poste = max(medianes.get(poste, 0.1), 0.05)
    impact = min(ga_per_90 / (2 * mediane_poste), 1.0)

    # 3. Régularité
    regularite = min(ligne["Starts"] / mp, 1.0)

    score = (poids["w_temps"] * temps_jeu
           + poids["w_impact"] * impact
           + poids["w_regularite"] * regularite)

    return float(score)


def calculer_score_cible(df_t, df_stats, medianes):
    index = construire_index_joueur(df_stats)
    
    scores = []
    
    for _, t in df_t.iterrows():
        joueur = t["player"]
        saison = t["season"]
        nouveau_club = t["to"]
        
        # Toutes les saisons connues du joueur
        saisons_joueur = index.get(joueur, {})
        
        # Stats S et S+1 !! (pas S+1 et S+2)
        stats_s1_par_team = saisons_joueur.get(saison, {})
        stats_s2_par_team = saisons_joueur.get(saison + 1, {})
        
        if nouveau_club in stats_s1_par_team:
            score_s1 = score_saison(stats_s1_par_team[nouveau_club], medianes)
            
            if nouveau_club in stats_s2_par_team:
                score_s2 = score_saison(stats_s2_par_team[nouveau_club], medianes)
                scores.append(0.6 * score_s1 + 0.4 * score_s2)
            elif stats_s2_par_team:
                scores.append(0.85 * score_s1)
            else:
                scores.append(0.7 * score_s1)
        
        elif stats_s1_par_team:
            # Pas dans le nouveau club, mais joue ailleurs en big 5
            scores_ailleurs = [score_saison(s, medianes) for s in stats_s1_par_team.values()]
            scores.append(0.4 * max(scores_ailleurs))
        
        else:
            # Disparu des big 5
            scores.append(0.10)
    
    return np.array(scores, dtype=np.float32)

#construction features

def construire_index_joueur(df_s):
    """
    Construit un index hiérarchique :
        index[player][season][team] -> dict des stats
    
    Permet de récupérer rapidement :
    - toutes les saisons d'un joueur : index[Player].keys()
    - une saison précise : index[Player][saison]
    - une saison dans un club précis : index[Player][saison][team]
    """
    index = {}
    
    for _, r in df_s.iterrows(): #r contient les rows (permat d'itérer sur le df)
        joueur = r["player"]
        saison = r["season"]
        team = r["team"]
        
        stats = {
            "team": team, "Classement":r["Classement"], "Premier League":r["Premier League"], "Liga":r["Liga"], "Serie A":r["Serie A"], "Bundesliga":r["Bundesliga"], "Ligue 1":r["Ligue 1"],
            "MP": r["MP"], "Min": r["Min"], "Starts": r["Starts"],
            "Subs": r["Subs"],"unSub": r["unSub"], "Gls": r["Gls"], "Ast": r["Ast"],
            "G-PK": r["G-PK"], "PK": r["PK"], "PKatt": r["PKatt"],
            "age": r["age"], "Pos_main": r["Pos_main"],
        }
        
        index.setdefault(joueur, {}).setdefault(saison, {})[team] = stats #permet de créer des dictionnaires imbriqués
    
    return index

def construire_sequence_transfert(transfert, index):
    """
    Construit la séquence pré-transfert pour un seul transfert.
    
    transfert : ligne du DataFrame transferts (Series)
    index : dict hiérarchique des stats
    
    Retourne :
        sequence : np.ndarray de forme (N_SAISONS_AVANT, n_features)
        masque : np.ndarray de forme (N_SAISONS_AVANT,) avec 0/1
    """
    n_stats_joueur = len(config.FEATURES_STATS)
    n_stats_club = len(config.FEATURES_C)
    n_postes = len(config.POSTES)
    n_features = n_stats_joueur + n_stats_club + n_postes + 2 # stats + postes + masque + age
    
    sequence = np.zeros((config.N_SAISONS_AVANT, n_features), dtype=np.float32)
    masque = np.zeros(config.N_SAISONS_AVANT, dtype=np.float32)
    
    joueur = transfert["player"]
    saison_t = transfert["season"]
    saisons_joueur = index.get(joueur, {})
    
    #attention : fonctionne pour les transferts d'été ET hiver
    for k in range(config.N_SAISONS_AVANT):
        saison_k = saison_t - config.N_SAISONS_AVANT + k
        stats_par_team = saisons_joueur.get(saison_k, {})
        
        if not stats_par_team:
            continue  # padding (zéros), masque reste à 0
        
        # Si plusieurs équipes cette saison : agréger les stats
        # On somme les stats numériques, on prend le poste majoritaire et l'âge max
        ligne = agreger_stats_multi_clubs(stats_par_team)
        
        # Remplir le vecteur de features
        vecteur = []
        for stat in config.FEATURES_STATS:
            v = ligne.get(stat, 0)
            vecteur.append(0.0 if v is None or pd.isna(v) else float(v))
        
        for stat in config.FEATURES_C:
            v = ligne.get(stat, 0)
            vecteur.append(0.0 if v is None or pd.isna(v) else float(v))
        
        age = ligne.get("age", 0)
        vecteur.append(0.0 if age is None or pd.isna(age) else float(age))
        
        poste = ligne.get("Pos_main", "")
        for p in config.POSTES:
            vecteur.append(1.0 if poste == p else 0.0) #on fait le one-hot !
        
        vecteur.append(1.0)  # masque local (la dernière feature)
        
        sequence[k] = np.array(vecteur, dtype=np.float32)
        masque[k] = 1.0
    
    return sequence, masque


def agreger_stats_multi_clubs(stats_par_team):
    """
    Si un joueur a joué dans plusieurs clubs une saison, on utilise UNIQUEMENT
    la ligne 'X Teams' qui contient déjà l'agrégation faite par FBref.
    
    Si pas de ligne 'X Teams' (cas mono-club), on retourne la seule ligne dispo.
    """
    # Chercher s'il existe une ligne 'X Teams'
    lignes_agregees = {
        team: stats for team, stats in stats_par_team.items()
        if "Teams" in team
    }
    
    if lignes_agregees:
        # On prend UNIQUEMENT la ligne agrégée (ignore les lignes par club)
        return list(lignes_agregees.values())[0]
    
    # Pas de ligne 'X Teams' : on retourne la seule ligne dispo
    if len(stats_par_team) == 1:
        return list(stats_par_team.values())[0]
    
    # Si plusieurs lignes par club mais pas de 'X Teams' → on agrège
    agg = {}
    for stat in config.FEATURES_STATS:
        agg[stat] = sum(
            s.get(stat, 0) or 0 
            for s in stats_par_team.values()
        )
    
    for stat in config.FEATURES_C:
        agg[stat] = sum(
            s.get(stat, 0) or 0 
            for s in stats_par_team.values()
        )
    
    ages = [s.get("age") for s in stats_par_team.values() if s.get("age")]
    agg["age"] = max(ages) if ages else 0
    
    postes = [s.get("Pos_main") for s in stats_par_team.values() if s.get("Pos_main")]
    agg["Pos_main"] = max(set(postes), key=postes.count) if postes else ""

    Classements = [s.get("Classement") for s in stats_par_team.values() if s.get("Classement")]
    agg["Classement"] = max(set(Classements), key=Classements.count) if Classements else ""
    
    return agg



def construire_toutes_sequences(df_t,index):
    """
    Construit la matrice (n_transferts, N_SAISONS_AVANT, n_features) 
    et la matrice de masques (n_transferts, N_SAISONS_AVANT).
    """
    sequences = []
    masques = []
    
    for _, transfert in df_t.iterrows():
        seq, mask = construire_sequence_transfert(transfert, index)
        sequences.append(seq)
        masques.append(mask)
    
    #on transforme la liste de tableaux en 1 tableau multidimensionel !
    X_seq = np.stack(sequences)  # forme (n, N_SAISONS_AVANT, n_features)
    masques = np.stack(masques)  # forme (n, N_SAISONS_AVANT)

    #on supprime les transferts sans historique (il y en a 834/5022 !!, c'est énorme et ça diminue bien nos données)
    #-> finalement y'en a moins en prenant la bonne saison !
    mask_valides = masques.sum(axis=1) > 0
    X_seq = X_seq[mask_valides]
    masques = masques[mask_valides]
    df_t_clean = df_t.reset_index(drop=True)[mask_valides].reset_index(drop=True)
    
    return X_seq, masques, df_t_clean

def construire_contexte(df_t):
    ctx = []
    for _, r in df_t.iterrows():
        sequence = [
            np.log1p(r["transfer_fee"]),
            np.log1p(r["market_val"])] #ensuite, je pourrais rajouter classement actuel club arrivée et club départ (et la ligue) pour plus de contexte
        ctx.append(sequence)
    return np.array(ctx, dtype=np.float32)

def split_temporel(df_t, X_seq, X_ctx, y, masques):
    """
    Sépare les données en train/val/test selon la saison du transfert.
    Pas de shuffle — l'ordre temporel est crucial pour éviter les fuites.
    
    Retourne un dict avec les 3 sous-ensembles, chacun étant
    un tuple (X_seq, X_ctx, y, masques).
    """
    saisons = df_t["season"].values
    
    mask_train = saisons <= 2019
    mask_val   = (saisons >= 2020) & (saisons <= 2021)
    mask_test  = (saisons >= 2022) & (saisons <= 2023)
    
    print(f"Split temporel :")
    print(f"  Train [≤2019] : {mask_train.sum()}")
    print(f"  Val   [2020-2021] : {mask_val.sum()}")
    print(f"  Test  [2022-2023] : {mask_test.sum()}")
    
    return {
        "train": (X_seq[mask_train], X_ctx[mask_train], y[mask_train]),
        "val":   (X_seq[mask_val],   X_ctx[mask_val],   y[mask_val]),
        "test":  (X_seq[mask_test],  X_ctx[mask_test],  y[mask_test]),
    }

def standardiser(X_train, X_val, X_test, n_features_a_norm):
    """
    Standardise les n_features_a_norm premières colonnes de chaque tableau.
    Le fit (calcul de la moyenne/écart-type) se fait sur le train UNIQUEMENT,
    puis la même transformation est appliquée à val et test.
    
    Fonctionne pour :
    - Tableaux 3D (séquences) : forme (n, N_SAISONS, n_features)
    - Tableaux 2D (contexte)  : forme (n, n_features)
    
    Les colonnes au-delà de n_features_a_norm (one-hot, masque...) sont 
    laissées intactes.
    """
    # Pour les séquences 3D, on aplatit pour calculer les stats globales
    if X_train.ndim == 3:
        X_flat = X_train[..., :n_features_a_norm].reshape(-1, n_features_a_norm)
    else:  # 2D
        X_flat = X_train[:, :n_features_a_norm]
    
    # Calculer moyenne et écart-type sur le train
    moyenne = X_flat.mean(axis=0)
    ecart_type = X_flat.std(axis=0) + 1e-8  # éviter division par 0
    
    # Appliquer la même transformation aux trois jeux
    for X in (X_train, X_val, X_test):
        X[..., :n_features_a_norm] = (X[..., :n_features_a_norm] - moyenne) / ecart_type
    
    return X_train, X_val, X_test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import charger_stats_joueurs,charger_transferts
    
    df_s = charger_stats_joueurs()
    df_t = charger_transferts()

    index = construire_index_joueur(df_s)
    medianes = calculer_medianes_par_poste(df_s)

    y = calculer_score_cible(df_t, df_s, medianes)
    print(y)
    print(y.shape)
    
    '''X_seq, masques = construire_toutes_sequences(df_t,index)
    print("X_seq :",X_seq)
    print("masques :",masques)
    print(X_seq.shape)
    print(masques.shape)


    # Certains joueurs on [0,0,0] comme masque. comptons-les
    # Compter les transferts sans aucune donnée
    n_vides = (masques.sum(axis=1) == 0).sum()
    print(f"Transferts sans aucun historique : {n_vides}/{len(masques)}")

    # Inspecter quelques cas
    df_t_reset = df_t.reset_index(drop=True)
    indices_vides = np.where(masques.sum(axis=1) == 0)[0][:5]
    for i in indices_vides:
        t = df_t_reset.iloc[i]
        print(f"\nJoueur: {t['player']!r} | Saison transfert: {t['season']!r} (type: {type(t['season']).__name__})")
        saisons_dispo = list(index.get(t['player'], {}).keys())
        print(f"  Saisons dispo dans index: {saisons_dispo}")
        if saisons_dispo:
            print(f"  Type saison index: {type(saisons_dispo[0]).__name__}")

    # Distribution des patterns de masque
    from collections import Counter
    patterns = Counter(tuple(m) for m in masques)
    print("Patterns de masque :")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    medianes = calculer_medianes_par_poste(df_s)
    print(f"Médianes (G+A)/90 par poste : {medianes}")
    
    # Tester sur quelques joueurs
    print("\n=== Test du score sur 5 joueurs ===")
    echantillon = df_stats[(df_stats["player"] == "Kylian Mbappé")].head(5)
    for _, ligne in echantillon.iterrows():
        s = score_saison(ligne, medianes)
        print(f"  {ligne['player']:25s} ({ligne['Pos_main']}, "
              f"saison {ligne['Saison']}, {ligne['Team']}) : "
              f"{ligne['Min']:.0f}min, {ligne['Ast']:.0f} Assist, "
              f"{ligne['Gls']:.0f} Goals → score = {s:.3f}")
        
    
    scores, categories = calculer_score_cible(df_transferts, df_stats, medianes)
    print("scores :",scores)
    print("catégories :", categories)

    recap = pd.DataFrame({
    "joueur": df_transferts["player"],
    "saison": df_transferts["Saison"],
    "from_club": df_transferts["from_FBREF"],
    "to_club": df_transferts["to_FBREF"],
    "categorie": categories,
    "score": scores.round(3),
    })

    print("\n=== 10 premiers transferts ===")
    print(recap.head(10).to_string(index=False))

    print("\n=== 10 transferts avec les meilleurs scores ===")
    print(recap.nlargest(40, "score").to_string(index=False))

    print("\n=== 10 transferts avec les pires scores (hors disparus) ===")
    print(recap[(recap["categorie"] != "disparu_des_big5")]
      .nsmallest(40, "score").to_string(index=False))


    
    index = construire_index_joueur(df_s)
    print("Construction des séquences...")
    X_seq, masques = construire_toutes_sequences(df_t, index)
    print(f"  X_seq : {X_seq.shape}")
    print(f"  masques : {masques.shape}")
    
    # Stats sur les masques
    n_saisons_par_transfert = masques.sum(axis=1)
    print(f"\n  Distribution du nombre de saisons d'historique :")
    for n in [0, 1, 2, 3]:
        count = (n_saisons_par_transfert == n).sum()
        pct = count / len(masques) * 100
        print(f"    {int(n)} saison(s) : {count:5d} ({pct:.1f}%)")
    
    # Inspecter un exemple
    print(f"\n  Exemple — séquence du transfert 0 ({df_t.iloc[0]['player']}) :")
    print(f"    Masque : {masques[0]}")
    print(f"    Première saison : {X_seq[0, 0]}")
    print(f"    Dernière saison : {X_seq[0, -1]}")

    print("\n=== DIAGNOSTIC SADIO MANÉ ===")
    mane_transfert = df_t[df_t["player"].str.contains("Mané", case=False, na=False)]
    print("Transferts de Mané dans la base :")
    print(mane_transfert[["player", "season", "from", "to"]].to_string())

    print("\nStats de Mané dans l'index :")
    saisons_mane = index.get("Sadio Mané", {})
    for s in sorted(saisons_mane.keys()):
        teams = list(saisons_mane[s].keys())
        print(f"  Saison {s} : équipes = {teams}")
        for team, stats in saisons_mane[s].items():
            print(f"    {team:25s} | MP={stats['MP']} | Min={stats['Min']} | "
                f"Gls={stats['Gls']} | Ast={stats['Ast']}")'''

        
