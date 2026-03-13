"""GPT-style Transformer for In-Context Learning regression tasks.

Architecture follows Garg et al. (2022) / von Oswald et al. (2023):
- Input: sequence of (x_i, y_i) tokens where x_i in R^p, y_i in R
- Last token is (x_q, 0) for the query
- Output: scalar prediction y_q from the query token position
- Full (non-causal) attention, pre-norm transformer layers
"""

import math
import torch
import torch.nn as nn


class ICLTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=12,
                 dropout=0.0, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward pass. x: (B, N, input_dim) -> pred: (B,)"""
        B, N, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(N, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.readout(h[:, -1, :]).squeeze(-1)

    def forward_with_intermediates(self, x):
        """Forward pass returning per-layer activations at the query position.

        Returns:
            pred: (B,) final predictions
            intermediates: list of (B, d_model) tensors, len = num_layers + 1
                intermediates[0] = after input projection
                intermediates[l] = after transformer layer l (1-indexed)
        """
        B, N, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(N, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos)

        intermediates = [h[:, -1, :].detach()]
        for layer in self.layers:
            h = layer(h)
            intermediates.append(h[:, -1, :].detach())

        h = self.final_norm(h)
        pred = self.readout(h[:, -1, :]).squeeze(-1)
        return pred, intermediates

    def predict_per_layer(self, x):
        """Get predictions at each layer using per-layer readout heads.

        Must call train_per_layer_readouts() first. Returns list of (B,) tensors.
        """
        B, N, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(N, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos)

        preds = []
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if hasattr(self, '_layer_readouts'):
                normed = self.final_norm(h)
                preds.append(self._layer_readouts[i](normed[:, -1, :]).squeeze(-1))
        return preds
