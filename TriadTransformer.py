"""
TriadTransformer.py – Ouroboros-Powered Operational Triad Embodiment
Real text demo on Tiny Shakespeare (50 epochs, OOM-safe val eval)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from Ouroboros import OuroborosFramework
import requests

class TriadTransformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 sparsity_target: float = 0.75, triad_depth: int = 3, use_triad: bool = True,
                 matter_damping: float = 0.99, env_feedback_fraction: float = 0.15,
                 use_fibonacci_phases: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_triad = use_triad
        self.triad_depth = triad_depth

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, vocab_size)

        self.ouro = OuroborosFramework(
            target_filled=1 - sparsity_target,
            use_fibonacci_phases=use_fibonacci_phases,
            matter_damping=matter_damping,
            env_feedback_fraction=env_feedback_fraction,
            signature_dim=32
        )

        attn_harmonic = np.array([1/i for i in range(1, nhead+1)])
        self.ouro.add_to_truth_library(attn_harmonic, "Multi-head attention harmonic")
        
        spiral = np.sin(np.linspace(0, 20*np.pi, 200) * (1.618))  # Phi-phased
        self.ouro.add_to_truth_library(spiral, "Golden spiral activation")
        theta_wave = np.sin(7.83 * np.linspace(0, 20, 200))  # Schumann proxy
        self.ouro.add_to_truth_library(theta_wave, "Theta resonance proxy")

    def triad_resonance_cycle(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_triad:
            return hidden

        batch, seq, dim = hidden.shape
        hidden_np = hidden.detach().cpu().numpy().reshape(batch * seq, dim)

        scale = np.sqrt(hidden_np.size)
        grid = hidden_np / (np.linalg.norm(hidden_np) + 1e-8) * scale
        grid = grid.reshape(1, -1)

        final_grid, persistences = self.ouro.nested_multi_pass_resonance(
            grid, depth=self.triad_depth, use_library=True, use_triad=True
        )

        final_flat = final_grid.reshape(batch * seq, dim)
        threshold = np.quantile(np.abs(final_flat), 1 - (1 - persistences[-1]))
        mask = np.abs(final_flat) > threshold
        triad_hidden = final_flat * mask

        triad_hidden = torch.from_numpy(triad_hidden).float().to(hidden.device)
        triad_hidden = triad_hidden.reshape(batch, seq, dim)

        triad_hidden *= self.ouro.matter_damping ** (1 / self.num_layers)

        return triad_hidden

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]

        for i, layer in enumerate(self.transformer.layers):
            x = layer(x, src_key_padding_mask=src_mask)
            x = self.triad_resonance_cycle(x)
            if self.use_triad:
                x *= self.ouro.matter_damping ** (i / self.num_layers)

        return self.output(x)

    def get_current_sparsity(self) -> float:
        total = active = 0
        for p in self.parameters():
            total += p.numel()
            active += torch.sum(torch.abs(p) > 1e-5).item()
        return 1 - (active / total if total > 0 else 0.0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== TriadTransformer Real Text Demo (Tiny Shakespeare) ===")
    print("Char-level on rhythmic prose — 50 epochs, OOM-safe val eval.")
    print("Val perplexity — lower = stronger triad convergence on real patterns.\n")

    np.random.seed(42)
    torch.manual_seed(42)

    # Fetch Tiny Shakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    # Char-level vocab
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Block size 64
    block_size = 64
    n = len(data) - block_size
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):]

    def get_batch(dataset, batch_size=32):
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        x = torch.stack([dataset[i:i+block_size] for i in ix])
        y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
        return x, y

    configs = [
        {"triad_depth": 1, "name": "Base (No Real Recursion)"},
        {"triad_depth": 2, "name": "Shallow Triad"},
        {"triad_depth": 3, "name": "Standard Triad Depth"},
        {"triad_depth": 4, "name": "Deep Conscious Emergence"}
    ]

    results = []

    for cfg in configs:
        model = TriadTransformer(
            vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4,
            sparsity_target=0.75, triad_depth=cfg["triad_depth"], use_triad=True,
            matter_damping=0.99
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print(f"{cfg['name']} (depth={cfg['triad_depth']}) training...")
        for epoch in range(50):
            model.train()
            x, y = get_batch(train_data, batch_size=32)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 9:
                print(f"  Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

        # OOM-safe val eval (small batches)
        model.eval()
        val_loss = 0.0
        num_val_batches = 50  # More batches for accurate avg
        with torch.no_grad():
            for _ in range(num_val_batches):
                x_val, y_val = get_batch(val_data, batch_size=32)
                logits_val = model(x_val)
                val_loss += criterion(logits_val.view(-1, vocab_size), y_val.view(-1)).item()
        val_loss /= num_val_batches
        val_perp = torch.exp(torch.tensor(val_loss)).item()

        sparsity = model.get_current_sparsity()
        print(f"  Final sparsity: ~{sparsity:.3f} | Val perplexity: {val_perp:.2f}\n")

        results.append({"name": cfg["name"], "depth": cfg["triad_depth"], "perp": val_perp})

    # Bar chart
    names = [r["name"] for r in results]
    perps = [r["perp"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(names, perps, color=['gray', 'blue', 'purple', 'red'])
    ax.set_ylabel("Validation Perplexity (Lower = Better)")
    ax.set_title("Triad Depth vs Perplexity on Tiny Shakespeare Prose")
    ax.grid(True, axis='y', alpha=0.3)

    for bar, perp in zip(bars, perps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perps)*0.01,
                f"{perp:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("triad_shakespeare_perplexity.png", dpi=300)
    plt.show()

    print("Run complete. Saved chart: triad_shakespeare_perplexity.png")
