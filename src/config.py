"""
Configuration centralisée du projet.
Tous les hyperparamètres au même endroit, pour expérimenter facilement.
"""

# --- Séquences pré-transfert ---
N_SAISONS_AVANT = 3                    # nombre de saisons d'historique en entrée

# Stats numériques utilisées dans les séquences
FEATURES_STATS = ["MP", "Min", "Starts", "Subs", "unSub", "Gls", "Ast", "G-PK","PK","PKatt"]
FEATURES_C = ["Classement","Liga", "Premier League", "Ligue 1", "Bundesliga", "Serie A"]

# Postes pour le one-hot
POSTES = ["GK", "DF", "MF", "FW"]

# --- Score cible ---
W_TEMPS = 0.55
W_IMPACT = 0.15
W_REGULARITE = 0.3
