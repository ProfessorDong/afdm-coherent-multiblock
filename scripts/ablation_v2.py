"""Regenerate Table IV (ablation study) with multi-seed error bars.

5 configurations at (P=5, N_p=32), B=4, 15 dB:
  Full MB-IDAR (default n_outer=6, n_lm_per_outer=3, rho_min=0.5, use_reacq=True)
  Without data-aided re-acquisition (use_reacq=False)
  Without safeguarded gradient refinement (n_lm_per_outer=0)
  Without reliability weighting (rho_min=0.0)
  Only pilot-only initial (n_outer=0)

Protocol: K=3 seeds x 8 batches x 16 realizations per configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver


SNR = 15.0
B_BLOCK = 4
N_SEEDS = 5
N_BATCHES = 8
BATCH_SIZE = 32


def eval_one_seed(cfg, seed, **kwargs):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                       constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                   batch_size=BATCH_SIZE, snr_db=SNR, generator=gen)
        with torch.no_grad():
            hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg, **kwargs)
        mask = batch.pilot_mask
        ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / N_BATCHES


def multi_seed(cfg, **kwargs):
    vals = [eval_one_seed(cfg, seed=k * 137 + 42, **kwargs) for k in range(N_SEEDS)]
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                            P=5, N_p=32, P_max=8)
    print(f"\n{'='*80}\nABLATION @ (P=5, N_p=32), B={B_BLOCK}, {SNR} dB\n"
          f"K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations\n{'='*80}")
    print(f"{'Configuration':<45s}  {'mean SER':>12s}  {'std':>10s}")

    ablation_configs = [
        ("Full MB-IDAR",
         dict(n_outer=6, n_lm_per_outer=3, rho_min=0.5, use_reacq=True)),
        ("without data-aided re-acquisition",
         dict(n_outer=6, n_lm_per_outer=3, rho_min=0.5, use_reacq=False)),
        ("without safeguarded gradient refinement",
         dict(n_outer=6, n_lm_per_outer=0, rho_min=0.5, use_reacq=True)),
        ("without reliability weighting",
         dict(n_outer=6, n_lm_per_outer=3, rho_min=0.0, use_reacq=True)),
        ("only pilot-only initial (no outer iters)",
         dict(n_outer=0, n_lm_per_outer=0, rho_min=0.5, use_reacq=False)),
    ]

    results = {}
    for name, kwargs in ablation_configs:
        t0 = time.time()
        m, s = multi_seed(cfg, **kwargs)
        dt = time.time() - t0
        print(f"{name:<45s}  {m:>12.4e}  {s:>10.4e}   ({dt:.0f}s)")
        results[name] = {"mean": m, "std": s, "kwargs": kwargs}

    out_path = Path("runs/ablation_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "B": B_BLOCK, "P": cfg.P, "N_p": cfg.N_p,
                   "N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
