"""Convergence of the outer loop, regenerated from the ACTUAL calibrated receiver.

convergence_v2.py duplicated the receiver inline and therefore predated the
v_eff calibration, producing SER values that no longer match Table II. This
script instead calls multiblock_dasbl_receiver directly with n_outer = T, under
the same protocol as Table II (5 seeds x 8 batches x 32), so that the T=6 point
is the Table II entry by construction.
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
T_LIST = [1, 2, 3, 4, 5, 6, 8, 10]
N_SEEDS = 5
N_BATCHES = 8
BATCH_SIZE = 32


def ser_at_T(cfg, T, seed):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                      constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    acc = 0.0
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=BATCH_SIZE, snr_db=SNR, generator=gen)
        with torch.no_grad():
            hard, _, _, _ = multiblock_dasbl_receiver(
                system, batch, const, cfg,
                n_outer=T, n_lm_per_outer=3, rho_min=0.5, use_reacq=True,
            )
        mask = batch.pilot_mask
        acc += float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
    return acc / N_BATCHES


def main():
    configs = [
        ("Easy", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("Hard", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ]
    out = {"snr_db": SNR, "B": B_BLOCK, "N_seeds": N_SEEDS,
           "N_batches": N_BATCHES, "batch_size": BATCH_SIZE, "results": {}}
    for name, cfg in configs:
        print(f"\n[{name}]  T   SER        std")
        per_T = {}
        for T in T_LIST:
            t0 = time.time()
            vals = [ser_at_T(cfg, T, k * 137 + 42) for k in range(N_SEEDS)]
            a = np.array(vals)
            per_T[str(T)] = {"mean": float(a.mean()), "std": float(a.std())}
            print(f"       {T:<3d} {a.mean()*100:7.2f}%   {a.std()*100:5.2f}   ({time.time()-t0:.0f}s)",
                  flush=True)
        out["results"][name] = per_T
    p = Path("runs/convergence_v3.json"); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
