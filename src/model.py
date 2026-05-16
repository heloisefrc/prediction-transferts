
"""
Architecture : LSTM sur séquence pré-transfert + side-input contexte.
"""
import torch
import torch.nn as nn
import config


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out : (batch, N_saisons, hidden_size)
        scores = self.attn(lstm_out)              # (batch, N_saisons, 1)
        weights = torch.softmax(scores, dim=1)    # (batch, N_saisons, 1)
        context = (weights * lstm_out).sum(dim=1) # (batch, hidden_size)
        return context


    
class TransferLSTM(nn.Module):
    def __init__(self, n_features_seq, n_features_ctx):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features_seq,
            hidden_size=config.HIDDEN_SIZE, #
            num_layers=config.NUM_LAYERS, #
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0, #
        )

        # AJOUT : Initialisation de la couche d'attention
        self.attention = SelfAttention(config.HIDDEN_SIZE) #

        self.ctx_encoder = nn.Sequential(
            nn.Linear(n_features_ctx, config.CONTEXT_EMBED_DIM), #
            nn.ReLU(),
            nn.Dropout(config.DROPOUT), #
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE + config.CONTEXT_EMBED_DIM, 32), #
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq, x_ctx):
        # x_seq: (batch, N_saisons, n_features_seq)
        
        # CORRECTION : On récupère lstm_out qui contient la séquence entière
        lstm_out, _ = self.lstm(x_seq) #
        
        # CORRECTION : On passe la séquence à la couche d'attention
        h_final = self.attention(lstm_out) #

        ctx_emb = self.ctx_encoder(x_ctx)
        combined = torch.cat([h_final, ctx_emb], dim=1)

        out = self.head(combined).squeeze(-1)
        return out
    

