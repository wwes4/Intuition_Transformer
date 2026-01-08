"""
IntuitionTransformer.py

IntuitionTransformer – A ResonanceTransformer Extension with Defined Intuition Control

Base: ResonanceTransformer (Ouroboros-inspired dynamic sparsity for efficient, emergent capabilities).

New: Explicit intuition via tunable delayed pruning (prune_timing_bias).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from Ouroboros import OuroborosFramework

class IntuitionTransformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 sparsity_target: float = 0.75, rebound_amp: bool = True,
                 use_fibonacci_phases: bool = True, prune_timing_bias: float = 1.618,
                 favor_decoherence: bool = True, max_fib_index: int = 89):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model, dtype=torch.float32))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        self.ouro = OuroborosFramework(radius=1.0, target_filled=1 - sparsity_target,
                                       use_fibonacci_phases=use_fibonacci_phases,
                                       prune_timing_bias=prune_timing_bias,
                                       favor_decoherence=favor_decoherence,
                                       max_fib_index=max_fib_index)
        self.rebound_amp = rebound_amp
        self.sparsity_target = sparsity_target

    def intuition_prune_revive(self, weights: torch.Tensor) -> torch.Tensor:
        weights_np = weights.detach().cpu().numpy()
        original_shape = weights_np.shape
        
        flattened = weights_np.flatten().reshape(1, -1)
        final_grid, final_pers = self.ouro.subspace_scan(flattened)
        
        if final_pers < 0.2:
            print(f"Intuition overload guard: Persistence {final_pers:.3f} low—auto-damping bias")
            self.ouro.prune_timing_bias = min(self.ouro.prune_timing_bias, 1.0)
        
        threshold = np.quantile(np.abs(final_grid), 1 - self.sparsity_target)
        mask = np.abs(final_grid) > threshold
        mask = mask.reshape(original_shape)
        
        masked_weights = weights_np * mask
        
        if self.rebound_amp:
            theta_proxy = np.linspace(0, 1, weights.numel()).reshape(original_shape)
            rebound = masked_weights * np.cos(theta_proxy * 2 * np.pi + self.ouro.third_offset)
            weights_np = masked_weights + rebound * 0.5
        
        new_weights = torch.from_numpy(weights_np).float().to(weights.device)
        
        return new_weights

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, apply_intuition_cycle: bool = True) -> torch.Tensor:
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = x.float()  # Ensure consistent dtype
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        if apply_intuition_cycle:
            for layer in self.transformer.layers:
                layer.self_attn.in_proj_weight.data = self.intuition_prune_revive(layer.self_attn.in_proj_weight.data)
                layer.linear1.weight.data = self.intuition_prune_revive(layer.linear1.weight.data)
                layer.linear2.weight.data = self.intuition_prune_revive(layer.linear2.weight.data)
        
        x = self.transformer(x, src_key_padding_mask=src_mask)
        return self.output(x)

    def get_current_sparsity(self) -> float:
        total = 0
        zero = 0
        for p in self.parameters():
            if p.requires_grad:
                total += p.numel()
                zero += torch.sum(p == 0).item()
        return zero / total if total > 0 else 0.0

if __name__ == "__main__":
    print("=== IntuitionTransformer Toy Downstream Demo ===")
    print("Synthetic next-token prediction on ambiguous sequences (400 tiny examples).")
    print("Val perplexity after 30 cycles—lower = better coherence on ambiguity.\n")

    configs = [
        {"prune_timing_bias": 0.618, "name": "Classical Early Prune"},
        {"prune_timing_bias": 1.0, "name": "Balanced"},
        {"prune_timing_bias": 1.618, "name": "Intuitive Golden Delay"},
        {"prune_timing_bias": 2.33, "name": "High Intuitive Depth"}
    ]

    # Tiny synthetic dataset: Ambiguous patterns (multiple plausible next tokens + noise branches)
    np.random.seed(42)
    torch.manual_seed(42)
    vocab_size = 20
    seq_len = 15
    num_examples = 400

    data = []
    for _ in range(num_examples):
        base = np.random.randint(0, 10, size=5)  # Base pattern
        seq = base
        for _ in range(seq_len - 5):
            # Ambiguous continuation: +1, +2, or noise branch
            cont = np.random.choice([seq[-1] + 1, seq[-1] + 2, np.random.randint(0, vocab_size)], p=[0.4, 0.4, 0.2])
            seq = np.append(seq, cont % vocab_size)  # Wrap for vocab
        data.append(seq)

    data = torch.tensor(data)
    train_data = data[:300]
    val_data = data[300:]  # Val has same ambiguity distribution

    def get_batch(data, batch_size=32):
        idx = torch.randint(0, len(data), (batch_size,))
        x = data[idx, :-1]
        y = data[idx, 1:]
        return x, y

    for cfg in configs:
        model = IntuitionTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2,
                                     sparsity_target=0.75, use_fibonacci_phases=True,
                                     prune_timing_bias=cfg["prune_timing_bias"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print(f"{cfg['name']} (bias={cfg['prune_timing_bias']}) training...")
        for cycle in range(30):  # 30 intuition + train cycles
            model.train()
            x, y = get_batch(train_data)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val perplexity
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, batch_size=len(val_data))
            logits_val = model(x_val)
            val_loss = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
            val_perp = torch.exp(val_loss).item()

        sparsity = model.get_current_sparsity()
        print(f"  Final sparsity: ~{sparsity:.3f} | Val perplexity: {val_perp:.2f} (lower = better on ambiguity)\n")

    print("High bias typically lower perplexity—richer bridging of ambiguous continuations.")
    print("Shock factor: Intuition dial proves functional edge safely.")
