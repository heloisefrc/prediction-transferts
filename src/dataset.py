"""
Dataset PyTorch : interface entre nos tableaux numpy et le modèle.
"""
import torch
from torch.utils.data import Dataset


class TransferDataset(Dataset):
    """
    Dataset qui contient les séquences pré-transfert, les contextes,
    et les scores cibles. À utiliser avec un DataLoader pour l'entraînement.
    """
    
    def __init__(self, X_seq, X_ctx, y):
        """
        X_seq : np.ndarray (n, N_SAISONS_AVANT, n_features_seq)
        X_ctx : np.ndarray (n, n_features_ctx)
        y     : np.ndarray (n,) — scores cibles dans [0, 1]
        """
        # Conversion en tenseurs PyTorch
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        """Nombre total d'exemples dans le dataset."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Retourne l'exemple à l'index idx, sous forme de tuple."""
        return self.X_seq[idx], self.X_ctx[idx], self.y[idx]