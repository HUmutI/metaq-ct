"""Q-Former-token conditioned Qwen report generator."""

from __future__ import annotations

import torch
from torch import nn


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(source)
        query = query + self.cross(q, kv, kv, need_weights=False)[0]
        return query + self.ff(self.ff_norm(query))


class CTTokenResampler(nn.Module):
    """Convert variable Q-Former banks to a matched fixed-size LM prefix."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024,
                 output_dim: int = 4096, num_queries: int = 32,
                 depth: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([ResamplerBlock(hidden_dim, heads, dropout)
                                     for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        source = self.input_proj(tokens)
        query = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, source)
        return self.out(query)
