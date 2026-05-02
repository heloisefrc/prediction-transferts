"""
Architecture : LSTM sur séquence pré-transfert + side-input contexte.
"""
import torch
import torch.nn as nn
import config


class TransferLSTM(nn.Module):
    """
    - Séquence des N saisons pré-transfert -> LSTM -> état caché final
    - Contexte du transfert (montant, valeur, âge, niveau clubs, etc.) -> Dense
    - Concaténation -> MLP -> score [0, 1]
    """

    def __init__(self, n_features_seq, n_features_ctx):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features_seq,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0,
        )

        self.ctx_encoder = nn.Sequential(
            nn.Linear(n_features_ctx, config.CONTEXT_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE + config.CONTEXT_EMBED_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # score dans [0, 1]
        )

    def forward(self, x_seq, x_ctx):
        # x_seq: (batch, N_saisons, n_features_seq)
        # x_ctx: (batch, n_features_ctx)
        _, (h_n, _) = self.lstm(x_seq)
        h_final = h_n[-1]  # dernière couche : (batch, hidden)

        ctx_emb = self.ctx_encoder(x_ctx)
        combined = torch.cat([h_final, ctx_emb], dim=1)

        out = self.head(combined).squeeze(-1)  # (batch,)
        return out
