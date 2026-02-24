#!/usr/bin/env python3
"""
Intent-Conditioned Flow Matching Model for aircraft trajectory prediction

This extends the original FlowMatchingModel by conditioning the generation
on explicit intent features derived from the input history's kinematic profile.

Architecture difference from the original:
    context_dim:  8  →  45  (8 original + 12 continuous intent + 25 one-hot)

Everything else (encoder depth, decoder depth, d_model, nhead) is identical.
This means the model has the same capacity to learn flow dynamics, but receives
a much richer signal about what the aircraft is doing
"""

import math
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ── Building blocks (same as original) ──────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)

class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, emb_dim: int = 128):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.linspace(0, math.log(10_000), emb_dim // 2, dtype=torch.float32)
            ),
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )

    def forward(self, t):
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        ang = t.unsqueeze(-1) * self.freqs
        temb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(temb)

class HistoryEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout, context_dim=8):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.context_proj = nn.Linear(context_dim, d_model) if context_dim > 0 else None
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        z = self.input(x)
        z = self.pos(z)
        if self.context_proj is not None and context is not None:
            c = self.context_proj(context).unsqueeze(1)
            z = torch.cat([c, z], dim=1)
        return self.norm(self.enc(z))

class FutureDenoiser(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=512)
        self.t_proj_tokens = nn.Linear(d_model, d_model)
        self.t_proj_memory = nn.Linear(d_model, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, in_dim)

    def forward(self, xt, mem, t_emb):
        z = self.pos(self.input(xt))
        z = z + self.t_proj_tokens(t_emb).unsqueeze(1)
        mem = mem + self.t_proj_memory(t_emb).unsqueeze(1)
        return self.output(self.norm(self.dec(tgt=z, memory=mem)))

# ── Main model ──────────────────────────────────────────────────────

class IntentFlowMatchingModel(nn.Module):
    """
    Intent-Conditioned Flow Matching Model.

    Identical architecture to FlowMatchingModel, except:
      - context_dim defaults to 45 (8 original + 12 intent continuous + 25 one-hot)
      - The HistoryEncoder.context_proj learns to map the enriched context

    The model is fully backwards-compatible: set context_dim=8 to recover
    the original model exactly.
    """

    # Default: 8 (original) + 12 (intent continuous) + 25 (one-hot) = 45
    ORIGINAL_CTX = 8
    INTENT_CONTINUOUS = 12
    INTENT_ONEHOT = 25
    DEFAULT_CTX = ORIGINAL_CTX + INTENT_CONTINUOUS + INTENT_ONEHOT

    def __init__(
        self,
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=8,
        ff=4 * 512,
        dropout=0.1,
        in_dim=7,
        context_dim=DEFAULT_CTX,   # <-- the key change
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.encoder = HistoryEncoder(
            in_dim, d_model, nhead, enc_layers, ff, dropout, context_dim=context_dim
        )
        self.time_emb = TimeEmbedding(d_model=d_model)
        self.denoiser = FutureDenoiser(in_dim, d_model, nhead, dec_layers, ff, dropout)

    def forward(self, x_hist, x_t, t_scalar, context):
        mem = self.encoder(x_hist, context)
        t_emb = self.time_emb(t_scalar)
        return self.denoiser(x_t, mem, t_emb)

# ── Sampling / ODE utilities ────────────────────────────────────────

def sample_xt_and_target(y, t):
    """Sample intermediate states and targets for flow matching training."""
    eps = torch.randn_like(y)
    t_ = t.view(-1, 1, 1)
    x_t = (1.0 - t_) * eps + t_ * y
    v_star = y - eps
    return x_t, v_star, eps

# ── Checkpoint management ───────────────────────────────────────────

def load_model_checkpoint(
    checkpoint_path: str, device=None) -> IntentFlowMatchingModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )
    cfg = ckpt.get("model_cfg", get_model_config())

    model = IntentFlowMatchingModel(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def get_model_config(context_dim: int = IntentFlowMatchingModel.DEFAULT_CTX) -> Dict[str, Any]:
    return dict(
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=8,
        ff=4 * 512,
        dropout=0.1,
        in_dim=7,
        context_dim=context_dim,
    )