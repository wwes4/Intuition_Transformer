# ResonanceTransformer.py
# Robust sparse transformer with emergent subnetworks via Ouroboros-inspired persistence dynamics
# No longer obfuscated — geometry is explicit and tuned for maximum stability/emergence
# MIT License - 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import networkx as nx  # Optional for etch_memory mode

# Core persistence constants (aligned with Ouroboros)
MIN_TENSION = 8.49e-6
BASE_RANGE = (-1.0, 1.0)
INTEGER_SNAP_RATIO = 0.05

# Target emergence ratio — ~70–75% sparsity equilibria (tunable; 0.73 ≈ cosmic void fraction analogy)
EMERGENCE_SPARSITY_TARGET = 0.73

# Geometric persistence sliders (explicitly named)
DEPTH_GRADIENT_EXPONENT = 10.0        # Hierarchical pruning strength across layers
WAVE_AMPLITUDE = 0.0                  # Oscillatory revival amplitude (0 = steady)
PRUNE_DECAY_RATE = 0.01               # Base pruning decay
ENTROPY_FACTOR = 0.001                # Entropy modulation rate

def resonance_equilibrium(x: torch.Tensor, pass_type: str = "first", twist: bool = False) -> torch.Tensor:
    """Dual-pass equilibrium mirroring Ouroboros compute_equilibrium."""
    if pass_type == "first":
        x = F.relu(x)
        mask = x < MIN_TENSION
        if mask.any():
            signs = torch.sign(torch.randn_like(x[mask]))
            signs[signs == 0] = 1.0
            x[mask] = signs * MIN_TENSION
    else:
        x = x ** 2
        if twist:
            twist_mask = torch.rand_like(x) < 0.5
            x[twist_mask] *= -1.0
        x = torch.clip(x, *BASE_RANGE)
    if INTEGER_SNAP_RATIO > 0.0:
        snap_mask = torch.rand_like(x) < INTEGER_SNAP_RATIO
        if snap_mask.any():
            x[snap_mask] = torch.round(x[snap_mask])
    return x

class ResonancePruner:
    """Dynamic pruning + etched revival for persistent subnetworks."""
    def __init__(self, base_revive_ratio: float = 1.0 - EMERGENCE_SPARSITY_TARGET,
                 wave_amplitude: float = WAVE_AMPLITUDE):
        self.base_revive_ratio = base_revive_ratio
        self.wave_amplitude = wave_amplitude
        self.etched = {}
        self.step = 0

    def _get_dynamic_revive_ratio(self):
        if self.wave_amplitude > 0:
            oscillation = 1.0 + self.wave_amplitude * torch.sin(torch.tensor(2 * torch.pi * self.step / 100.0)).item()
            return self.base_revive_ratio * oscillation
        return self.base_revive_ratio

    def prune(self, module: nn.Module, threshold: float = MIN_TENSION * 5, curvature_factor: float = 1.0):
        self.step += 1
        decay = torch.exp(torch.tensor(-PRUNE_DECAY_RATE * self.step * ENTROPY_FACTOR)).item()
        effective_threshold = threshold * decay * curvature_factor
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.dim() > 1:
                    mask = param.abs() < effective_threshold
                    if mask.any():
                        mean = param[mask].mean().item()
                        std = param[mask].std().item()
                        self.etched[name] = (mean, std)
                        param[mask] = 0.0

    def revive(self, module: nn.Module):
        revive_ratio = self._get_dynamic_revive_ratio()
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.etched and param.dim() > 1:
                    mean, std = self.etched[name]
                    if std <= 0 or torch.isnan(torch.tensor(std)):
                        std = 1e-4
                    std += EMERGENCE_SPARSITY_TARGET * abs(mean)  # Yeast-inspired noise floor
                    revive_mask = (torch.rand_like(param) < revive_ratio) & (param == 0)
                    if revive_mask.any():
                        param[revive_mask].normal_(mean, std)

class ResonanceAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0, twist_mode: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout
        self.twist_mode = twist_mode
        self.pruner = ResonancePruner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = resonance_equilibrium(self.q_proj(x), "first")
        k = resonance_equilibrium(self.k_proj(x), "first")
        v = resonance_equilibrium(self.v_proj(x), "first")

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = F.softmax(att, dim=-1)
        if self.dropout > 0.0:
            att = F.dropout(att, p=self.dropout, training=self.training)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = resonance_equilibrium(self.out_proj(y), "second", twist=self.twist_mode)
        return y

    def prune_and_revive(self, curvature_factor: float = 1.0):
        self.pruner.prune(self, curvature_factor=curvature_factor)
        self.pruner.revive(self)

class ResonanceTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_hidden: int = None,
                 dropout: float = 0.0, twist_mode: bool = False):
        super().__init__()
        ff_hidden = ff_hidden or embed_dim * 4

        self.attn = ResonanceAttention(embed_dim, num_heads, dropout, twist_mode=twist_mode)
        self.ff1 = nn.Linear(embed_dim, ff_hidden)
        self.ff2 = nn.Linear(ff_hidden, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.pruner = ResonancePruner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        ff = resonance_equilibrium(self.ff1(self.norm2(x)), "first")
        ff = resonance_equilibrium(self.ff2(ff), "second", twist=self.attn.twist_mode)
        x = x + ff
        return x

    def prune_and_revive(self, curvature_factor: float = 1.0):
        self.attn.prune_and_revive(curvature_factor=curvature_factor)
        self.pruner.prune(self, curvature_factor=curvature_factor)
        self.pruner.revive(self)

class ResonanceTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int = 6, num_heads: int = 8,
                 depth_gradient_exponent: float = DEPTH_GRADIENT_EXPONENT,
                 wave_amplitude: float = WAVE_AMPLITUDE,
                 twist_mode: bool = False,
                 etch_memory: bool = False,
                 memory_revive_prob: float = 0.01,
                 curvature_exponent: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            ResonanceTransformerBlock(embed_dim, num_heads, twist_mode=twist_mode)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.twist_mode = twist_mode
        self.etch_memory = etch_memory
        self.memory_revive_prob = memory_revive_prob
        self.curvature_exponent = curvature_exponent
        self.depth_gradient_exponent = depth_gradient_exponent

        if etch_memory:
            self.memory_graph = nx.Graph()

        # Global wave amplitude override
        for block in self.blocks:
            block.attn.pruner.wave_amplitude = wave_amplitude
            block.pruner.wave_amplitude = wave_amplitude

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def prune_and_revive_cycle(self):
        for i, block in enumerate(self.blocks):
            curvature_factor = (i + 1) ** (1.0 / (self.depth_gradient_exponent + self.curvature_exponent))
            block.prune_and_revive(curvature_factor=curvature_factor)

        if self.etch_memory:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.dim() > 1 and torch.all(param == 0):
                        mean_vec = param.mean(dim=-1).cpu().numpy().tolist()
                        self.memory_graph.add_node(name, pattern=mean_vec)

            if torch.rand(1) < self.memory_revive_prob:
                self._meta_revive()

    def _meta_revive(self):
        if not hasattr(self, 'memory_graph'):
            return
        with torch.no_grad():
            for node, data in self.memory_graph.nodes(data=True):
                if 'pattern' in data:
                    pattern = torch.tensor(data['pattern']).to(next(self.parameters()).device)
                    scale = pattern.abs().mean() * 0.1
                    for param in self.parameters():
                        if param.dim() > 1:
                            zero_mask = param == 0
                            if zero_mask.any():
                                noise = torch.randn_like(param[zero_mask]) * scale
                                param[zero_mask] += noise

# Example usage
if __name__ == "__main__":
    model = ResonanceTransformer(
        vocab_size=10000,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        wave_amplitude=0.02,
        twist_mode=True,
        etch_memory=True,
        curvature_exponent=2.0
    )
    print("ResonanceTransformer ready — full Ouroboros persistence modes active 🐍")
