
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #le chemin du projet (on peut lancer l'execution d'où on veut) --> Path(__file__) c'est le chemin de data_loader, et on le transforme en chemin absolu
DATA_PATH = BASE_DIR / "data" / "data.xlsx" 

def charger_transferts():
    """
    Charge et nettoie la feuille des transferts.
    Retourne un DataFrame avec les transferts utilisables pour l'entraînement.
    """
    # Chargement brut
    df = pd.read_excel(DATA_PATH, sheet_name="transfers")
    print(f"[transferts] {len(df)} lignes au chargement initial")

    # nettoyage
    # Filtrer les saisons exploitables
    df = df[(df["season"] >= 2007) & (df["season"] <= 2023)] #avant 2007, on manque de données avant, après 2023, on manque de données après
    print(f"[transferts] {len(df)} après filtre saisons [2007-2023]")

    #repérer transferts trève hivernale
    # Filtrer les transferts d'hiver (janvier-février)
    mois = df["date"].dt.month
    df["transfert_hiver"] = mois.isin([1, 2]).astype(int)

    # gestion valeurs manquantes
    df["transfer_fee"] = df["transfer_fee"].fillna(0)
    df["market_val"] = df["market_val"].fillna(df["market_val"].median())

    #supprimer lignes doublons
    df = df.drop_duplicates()
    print(f"[transferts] {len(df)} après supp doublons")

    #supprimer colonnes inutiles
    df = df.drop(columns=["player_id", "from_club_id","to_club_id"])

    #reset l'index
    df = df.reset_index(drop=True)



    return df

def charger_stats_joueurs():
    """
    Charge et nettoie la feuille 'season' 
    Retourne un DataFrame propre.
    """
    df = pd.read_excel(DATA_PATH, sheet_name="season")
    print(f"[stats] {len(df)} lignes au chargement initial")
    

    # convertir en numérique les colonnes de stats
    colonnes_numeriques = ["MP", "Min", "Starts", "Subs", "unSub", "Gls", "Ast", "G-PK", "PK", "PKatt", "age"]
    for col in colonnes_numeriques:
        df[col] = pd.to_numeric(df[col], errors="coerce") #si pas convertible : NaN


    # créer la colonne Pos_main (les 2 premières lettres de Pos)
    df["Pos_main"] = df["Pos"].fillna("").str[:2]
    df = df[df["Pos_main"].isin(["GK", "DF", "MF", "FW"])]
    print(f"[stats] {len(df)} lignes en supprimant les sans poste")


    # retirer les saisons où le joueur n'a pas joué (Min < 90) --> on verra ça plus tard je pense
    '''df = df.dropna(subset=["Min"])
    df = df[df["Min"] >= 90]'''

    #supprimer les colonnes qui se répètent
    df = df.drop(columns=["G+A", "90s","PKm","Pos","Rk"]) #colonnes redondantes avec d'autres


    
    df = df.reset_index(drop=True)
    print(f"[stats] {len(df)} saisons-joueurs au final\n")
    return df




if __name__ == "__main__": #se lance que quand on lance directement data_loader.py
    df_t = charger_transferts()
    df_s = charger_stats_joueurs()
    
    print("=== Aperçu transferts ===")
    print(df_t.head())
    
    print("\n=== Aperçu stats ===")
    print(df_s.head())
    print(f"\nSaisons stats : {df_s['season'].min()} → {df_s['season'].max()}")
    print(f"Postes : {df_s['Pos_main'].value_counts().to_dict()}")
