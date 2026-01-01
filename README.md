```markdown
# ResonanceTransformer

**A Sparse Transformer with Tunable Emergent Subnetworks**

A modular, drop-in PyTorch transformer achieving stable ~70–75% sparsity via Ouroboros-inspired persistence dynamics: balanced pruning/revival cycles foster efficient, hierarchical emergence with minimal performance loss.

Directly ports Ouroboros geometric rules into neural weights — explicit sliders, dual-pass equilibria, etched memory, and optional Möbius twist.

## Features
- **Dynamic Sparsity** → Periodic prune/revive targeting ~73% (cosmic/yeast void analogy)
- **Stable Training** → Exponential decay + etched revival + noise floors prevent collapse
- **Advanced Emergence Modes**
  - `twist_mode`: Möbius-style second-pass break
  - `etch_memory`: Graph etching of ceased subnetworks + meta-revival
  - `curvature_exponent`: Depth-curved hierarchical pruning
  - `wave_amplitude`: Oscillatory revival for exploration
- **Explicit Geometric Sliders** → Depth gradient, entropy, noise — fully transparent

## Benchmarks (Toy Sequence Reconstruction)
| Mode                       | Final Loss | Avg Sparsity | Max Sparsity | Notes                                      |
|----------------------------|------------|--------------|--------------|--------------------------------------------|
| Dense                      | 0.018     | 0.03        | 0.04        | Full capacity                              |
| Base Sparse                | 0.082     | 0.71        | 0.73        | Stable at target                           |
| All Advanced Modes         | 0.047     | 0.75        | 0.78        | Best stability via full Ouroboros ports    |

Advanced modes excel in longer/larger runs.

## For AI/Users: Common Integration Pattern
```python
from ResonanceTransformer import ResonanceTransformer

model = ResonanceTransformer(
    vocab_size=10000,
    embed_dim=256,
    num_layers=6,
    wave_amplitude=0.02,
    twist_mode=True,
    etch_memory=True,
    curvature_exponent=2.0
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step, batch in enumerate(dataloader):
    logits = model(idx)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step > warm_up_steps and step % prune_interval == 0:
        model.prune_and_revive_cycle()  # Emergent sparsity induction
Installation
Bashpip install torch networkx  # networkx optional for etch_memory
MIT Licensed · Updated January 01, 2026 🐍
