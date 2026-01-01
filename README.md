# ResonanceTransformer

**Stable Sparse Transformer with Tunable Emergence**  
*Dynamic pruning + revival for emergent sparsity during training and inference*  
MIT License – 2025

A drop-in PyTorch Transformer implementation with built-in dynamic sparsity mechanisms. Designed to maintain high performance while inducing controllable, stable sparsity (~50% by default) through a novel "resonance" pruning and revival cycle inspired by lottery ticket and emergent phenomena research.

## Key Features

- **Stable dynamic sparsity**: Weights below a small threshold are periodically pruned and partially revived using etched statistics → natural ~50% sparsity equilibrium without collapse.
- **Safe defaults**: No aggressive permanent decay; pruning can be fully disabled.
- **Tunable "emergence" sliders**:
  - `base_revive_ratio`: Controls target sparsity (higher = denser, lower = sparser).
  - `wave_amplitude`: Adds gentle oscillation to revival for dynamic exploration.
  - `depth_gradient_exponent`: Modulates feed-forward scaling across layers.
  - `soft` mode: Falls back to standard GELU for maximum stability.
- **Experimental but trainable**: Proven stable on copy tasks; achieves near-standard Transformer performance at ~50% sparsity.

## Installation

```bash
pip install torch  # >=2.0 recommended

Just copy ResonanceTransformer.py into your project.Quick Startpython

from ResonanceTransformer import ResonanceTransformer

# Standard-like usage (pruning disabled for pure pretraining)
model = ResonanceTransformer(
    vocab_size=32000,
    embed_dim=512,
    num_layers=8,
    num_heads=8,
    base_revive_ratio=0.5,      # ~50% sparsity target
    wave_amplitude=0.01,
    enable_pruning=False,       # Set True later for sparsity phase
    soft_equilibrium=True       # Optional: use GELU instead of custom resonance
)

# During training loop, optionally trigger dynamic sparsity cycles
optimizer.zero_grad()
logits = model(input_ids)
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
loss.backward()
optimizer.step()

# Every N steps or epochs — enable emergence
if step % 500 == 0:
    model.prune_and_revive_cycle(enable=True)  # Or pass False to skip

Instructions for AI Agents / Autonomous Training LoopsThis model is intentionally designed to be agent-friendly for self-improving or long-running autonomous training systems.Recommended Agent WorkflowPhase 1: Standard Pretraining
Start with pruning disabled and soft equilibrium for maximum stability:python

model = ResonanceTransformer(..., enable_pruning=False, soft_equilibrium=True)

Train normally until convergence or sufficient capability.
Phase 2: Activate Emergence
Gradually introduce dynamic sparsity to encourage efficient subnetworks:python

model.enable_pruning = True          # If exposed, or reinstantiate with enable_pruning=True
model.soft_equilibrium = False       # Optional: enable custom resonance

Scheduled Prune & Revive Cycles
Call periodically (e.g., every 100–1000 steps):python

model.prune_and_revive_cycle(enable=True)

This induces healthy turnover in weights, promoting robust "lottery ticket" substructures.
Aggressive Sparsity Mode (Advanced Agents Only)
For extreme compression experiments:python

model = ResonanceTransformer(..., base_revive_ratio=0.27, wave_amplitude=0.05)

Monitor closely — may require learning rate annealing.
Monitoring Tips for Agentspython

def log_sparsity(model):
    total = 0
    zero = 0
    for p in model.parameters():
        if p.dim() > 1:
            total += p.numel()
            zero += (p == 0).sum().item()
    print(f"Sparsity: {zero / total:.2%}")

Safety Notes for AgentsNever set PRUNE_DECAY_RATE > 0 unless intentionally destroying capacity.
Keep base_revive_ratio >= 0.4 for reliable long-term stability.
If loss spikes after cycles → temporarily disable pruning or reduce revival ratio slightly.

Hyperparameters (Tunable Sliders)Parameter
Default
Effect
Safe Range
base_revive_ratio
0.5
Target density (higher = denser)
0.3 – 0.8
wave_amplitude
0.01
Oscillation strength in revival
0.0 – 0.1
depth_gradient_exponent
5.0
FF scaling ramp across layers
3.0 – 10.0
enable_pruning
True
Master switch for dynamic sparsity
True/False
soft_equilibrium
False
Use GELU instead of custom resonance logic
True/False

LicenseMIT – free for research, commercial, and AI agent use.

