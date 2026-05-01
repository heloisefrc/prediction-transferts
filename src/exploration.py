import pandas as pd

CHEMIN = "../data/data.xlsx"
# Récupérer la liste des feuilles
'''fichier = pd.ExcelFile(CHEMIN)
print("Feuilles disponibles :")
for nom in fichier.sheet_names:
    print(nom)'''

df_transferts = pd.read_excel(CHEMIN, sheet_name="transfers")
'''print("\n=== Final_transferts_big_5 ===")
print(f"Forme : {df_transferts.shape}")  # (lignes, colonnes)
print(f"\nColonnes : {list(df_transferts.columns)}")
print(f"\nPremières lignes :")
print(df_transferts.head())'''

df_stats = pd.read_excel(CHEMIN, sheet_name="players_season")
'''print("\n=== players big 5 ===")
print(f"Forme : {df_stats.shape}")
print(f"\nColonnes : {list(df_stats.columns)}")
print(df_stats.head())

print("\n=== Types des colonnes (transferts) ===")
print(df_transferts.dtypes)

print("\n=== Valeurs manquantes (transferts) ===")
print(df_transferts.isna().sum())'''

'''doublons = df_stats.groupby(["Player", "Saison"]).size()
doublons = doublons[doublons > 1].sort_values(ascending=False)
print(f"\nNombre de cas (joueur, saison) avec >1 lignes : {len(doublons)}")
print("Top 5 :")
print(doublons.head())

transfert_test = df_transferts.iloc[0]
print(f"\n=== Vérification d'alignement des saisons ===")
print(f"Transfert : {transfert_test['player_name']}")
print(f"  Saison transfert : {transfert_test['Saison']}")
print(f"  Date transfert   : {transfert_test['transfer_date']}")
print(f"  De : {transfert_test['from_FBREF']} -> Vers : {transfert_test['to_FBREF']}")

# Cherchons toutes les lignes de stats de ce joueur
stats_joueur = df_stats[df_stats["Player"] == transfert_test["player_name"]]
print(f"\nLignes stats trouvées pour {transfert_test['player_name']} :")
print(stats_joueur[["Player", "Saison", "Team", "MP", "Min"]].to_string())

print("\n=== Répartition des transferts par saison ===")
print(df_transferts["Saison"].value_counts().sort_index())


doublons = df_stats.groupby(["Player", "Saison"]).size()
doublons = doublons[doublons > 1]
print(f"\nNombre de cas (joueur, saison) avec >1 lignes : {len(doublons)}")
print("Top 5 :")
print(doublons.sort_values(ascending=False).head())

# Cherchons des exemples de "2 teams", "3 teams"
exemples = df_stats[df_stats["Team"].str.contains("Teams", case=False, na=False)]
print(f"\n=== Lignes 'X teams' ===")
print(f"Nombre total : {len(exemples)}")
print(f"\nValeurs uniques de 'Team' contenant 'teams' :")
print(exemples["Team"].value_counts().head(10))

# Un exemple complet
print(f"\nExemple :")
print(exemples.head(3).to_string())

n_doublons_complets = df_transferts.duplicated().sum()
print(f"Lignes 100% identiques (toutes colonnes) : {n_doublons_complets}")

doublons = df_transferts[df_transferts.duplicated(subset=["player_id", "transfer_date", 
                                     "from_club_id", "to_club_id"], 
                            keep=False)]  # keep=False : garde TOUS les exemplaires
doublons = doublons.sort_values(["player_id", "transfer_date"])
print(f"\nTotal de lignes impliquées dans des doublons : {len(doublons)}")
print(f"\nExemples (10 premières lignes) :")
print(doublons.head(10).to_string())'''



# 1. Combien de lignes "X Teams" ?
mask_teams = df_stats["team"].str.contains("Teams", na=False)
print(f"Lignes 'X Teams' totales : {mask_teams.sum()}")
print(df_stats[mask_teams]["team"].value_counts())

# 2. Pour chaque joueur ayant une ligne "X Teams", regarder s'il a AUSSI 
# les lignes par club ou seulement la ligne agrégée
print("\n=== Diagnostic : 'X Teams' avec/sans lignes par club ===")

# Joueurs qui ont une ligne "X Teams" sur une saison donnée
joueurs_teams = df_stats[mask_teams][["player", "season"]].drop_duplicates()
print(f"Couples (joueur, saison) avec ligne 'X Teams' : {len(joueurs_teams)}")

avec_lignes_clubs = 0
sans_lignes_clubs = 0
for _, row in joueurs_teams.iterrows():
    sub = df_stats[
        (df_stats["player"] == row["player"]) 
        & (df_stats["season"] == row["season"])
    ]
    n_teams_lines = sub["team"].str.contains("Teams", na=False).sum()
    n_real_clubs = len(sub) - n_teams_lines
    
    if n_real_clubs > 0:
        avec_lignes_clubs += 1
    else:
        sans_lignes_clubs += 1

print(f"  Avec lignes par club aussi : {avec_lignes_clubs}")
print(f"  Avec UNIQUEMENT 'X Teams'  : {sans_lignes_clubs}")

# 3. Montrer 3 exemples de chaque cas
print("\n=== EXEMPLE : 'X Teams' AVEC lignes par club ===")
for _, row in joueurs_teams.head(20).iterrows():
    sub = df_stats[
        (df_stats["player"] == row["player"]) 
        & (df_stats["season"] == row["season"])
    ]
    n_teams = sub["team"].str.contains("Teams", na=False).sum()
    if len(sub) - n_teams > 0:  # cas avec lignes par club
        print(f"\n--- {row['player']}, saison {row['season']} ---")
        print(sub[["player", "season", "team", "MP", "Min", "Gls", "Ast"]].to_string(index=False))
        break

print("\n=== EXEMPLE : 'X Teams' SANS lignes par club ===")
for _, row in joueurs_teams.iterrows():
    sub = df_stats[
        (df_stats["player"] == row["player"]) 
        & (df_stats["season"] == row["season"])
    ]
    n_teams = sub["team"].str.contains("Teams", na=False).sum()
    if len(sub) - n_teams == 0:  # cas avec uniquement Teams
        print(f"\n--- {row['player']}, saison {row['season']} ---")
        print(sub[["player", "season", "team", "MP", "Min", "Gls", "Ast"]].to_string(index=False))
        break