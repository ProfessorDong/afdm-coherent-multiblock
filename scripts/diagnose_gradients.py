"""Diagnose gradient magnitudes across the network to find vanishing/exploding paths.

For each of three inits (default, v2, v2 with gate half-open), compute the
gradient of a real training batch loss and print per-parameter norm statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig
from afdm.loss import compose_training_loss
from afdm.training import TrainingConfig, sample_batch


def apply_v2_init(rx, gate_b, gamma_raw):
    for layer in rx.layers:
        layer.set_transformer.output_proj.weight.data.zero_()
        layer.set_transformer.output_proj.bias.data.zero_()
        layer.gate.b.data.fill_(gate_b)
        layer.gamma_raw.data.fill_(gamma_raw)
    return rx


@torch.enable_grad()
def compute_gradients(rx, cfg, tc, mu_ce=5.0):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
    batch = sample_batch(system, channel, const, pp, pv, batch_size=32, snr_db=15.0, generator=gen)
    out = rx(batch["r"], sigma_w2_block=batch["sigma_w2_block"], return_layer_states=True)
    loss, breakdown = compose_training_loss(
        out, batch["x_true"], batch["labels"], batch["h_true"],
        batch["theta_true"], batch["pilot_mask"],
        layer_gamma=0.7, mu_ce=mu_ce, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
    )
    rx.zero_grad(set_to_none=True)
    loss.backward()

    stats = {}
    for name, p in rx.named_parameters():
        if p.grad is None:
            stats[name] = (0, 0.0)
        else:
            stats[name] = (p.numel(), p.grad.norm().item() / max(p.numel(), 1) ** 0.5)
    return loss.item(), breakdown, stats


def summarize_group(stats, group_name, prefix_match):
    """Aggregate stats for parameters whose name contains prefix_match."""
    matches = [(name, n, g) for name, (n, g) in stats.items() if prefix_match in name]
    if not matches:
        return
    total_n = sum(n for _, n, _ in matches)
    max_g = max((g for _, _, g in matches), default=0.0)
    min_g = min((g for _, _, g in matches), default=0.0)
    mean_g = sum(g for _, _, g in matches) / max(len(matches), 1)
    print(f"    {group_name:<25s}: {len(matches)} tensors, {total_n:>8,} params, "
          f"grad-norm-per-param: min {min_g:.2e}, mean {mean_g:.2e}, max {max_g:.2e}")


def main():
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=3, seed=0,
    )
    tc = TrainingConfig(lr=5e-4, n_epochs=1, steps_per_epoch=1, batch_size=32,
                        val_snr_dbs=(15.0,), mu_ce=5.0)

    configs = [
        ("A. Default random init",       lambda rx: rx),
        ("B. v2 (gate closed, delta=0)", lambda rx: apply_v2_init(rx, gate_b=-8.0, gamma_raw=-5.0)),
        ("C. Half-open gate + delta=0",  lambda rx: apply_v2_init(rx, gate_b=0.0, gamma_raw=-5.0)),
        ("D. Open gate + random delta",  lambda rx: (
            [layer.gate.b.data.fill_(2.0) for layer in rx.layers] and rx
        )),
    ]

    for name, init_fn in configs:
        torch.manual_seed(0)
        rx = init_fn(cfg.receiver())
        loss, breakdown, stats = compute_gradients(rx, cfg, tc)
        print(f"\n{name}: loss={loss:.3f}  set={breakdown['set_loss']:.3f}  ce={breakdown['ce']:.3f}")
        summarize_group(stats, "alpha_raw",           "alpha_raw")
        summarize_group(stats, "lambda_raw",          "lambda_raw")
        summarize_group(stats, "sigma_calib_raw",     "sigma_calib_raw")
        summarize_group(stats, "omega_raw",           "omega_raw")
        summarize_group(stats, "gate.b",              "gate.b")
        summarize_group(stats, "gate.tilde_a",        "gate.tilde_a")
        summarize_group(stats, "SetTransformer.input_proj",  "set_transformer.input_proj")
        summarize_group(stats, "SetTransformer.output_proj", "set_transformer.output_proj")
        summarize_group(stats, "SetTransformer.blocks",      "set_transformer.blocks")


if __name__ == "__main__":
    main()
