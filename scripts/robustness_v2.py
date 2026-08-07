"""Noise-variance mismatch + QAM16 robustness under the physical model.

Two claims in the paper need backing runs:
  (a) sigma_w^2 mismatch: replace sigma_w2 with alpha * sigma_w2 in the receiver
      (affects softmax temperature omega and CG regularization) at HARD B=4, 15 dB.
  (b) QAM16 vs QPSK at (P=5, N_p=32), B=4, 15 dB.

Protocol: 3 seeds x 8 batches x 32 realizations.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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
N_SEEDS = 3
N_BATCHES = 8
BATCH_SIZE = 32


def eval_cfg(cfg, seed, sigma_scale=1.0):
    """Run MB-IDAR; if sigma_scale != 1, the receiver is fed a mismatched sigma_w2."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                      constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=BATCH_SIZE, snr_db=SNR, generator=gen)
        if sigma_scale != 1.0:
            # receiver believes a wrong noise variance
            batch = replace(batch, sigma_w2_block=batch.sigma_w2_block * sigma_scale)
        with torch.no_grad():
            hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                      n_outer=6, n_lm_per_outer=3,
                                                      rho_min=0.5, use_reacq=True)
        mask = batch.pilot_mask
        ser_acc += float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
    return ser_acc / N_BATCHES


def multi_seed(cfg, **kw):
    v = np.array([eval_cfg(cfg, seed=k * 137 + 42, **kw) for k in range(N_SEEDS)])
    return float(v.mean()), float(v.std())


def main():
    out = {}

    # (a) noise-variance mismatch at HARD (P=5, N_p=16), B=4  -- matches Table II row
    cfg_hard = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    print(f"\n{'='*72}\n(a) NOISE-VARIANCE MISMATCH @ HARD (P=5,N_p=16), B={B_BLOCK}, {SNR} dB"
          f"\n    K={N_SEEDS} seeds x {N_BATCHES} x {BATCH_SIZE}\n{'='*72}")
    print(f"{'alpha':>8s}  {'mean SER':>12s}  {'std':>10s}")
    mism = {}
    for alpha in (0.5, 1.0, 2.0, 4.0):
        t0 = time.time()
        m, s = multi_seed(cfg_hard, sigma_scale=alpha)
        print(f"{alpha:>8.1f}  {m:>12.4e}  {s:>10.4e}   ({time.time()-t0:.0f}s)")
        mism[str(alpha)] = {"mean": m, "std": s}
    out["noise_mismatch_hard_Np16"] = mism

    # (b) QAM16 vs QPSK at (P=5, N_p=32), B=4  -- matches Table IV config
    print(f"\n{'='*72}\n(b) CONSTELLATION @ (P=5,N_p=32), B={B_BLOCK}, {SNR} dB\n{'='*72}")
    print(f"{'constellation':>14s}  {'mean SER':>12s}  {'std':>10s}")
    consts = {}
    for kind in ("qpsk", "qam16"):
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32,
                               P_max=8, constellation_kind=kind)
        t0 = time.time()
        m, s = multi_seed(cfg)
        print(f"{kind:>14s}  {m:>12.4e}  {s:>10.4e}   ({time.time()-t0:.0f}s)")
        consts[kind] = {"mean": m, "std": s}
    out["constellation_P5_Np32"] = consts

    p = Path("runs/robustness_v2.json"); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"snr_db": SNR, "B": B_BLOCK, "N_seeds": N_SEEDS,
               "N_batches": N_BATCHES, "batch_size": BATCH_SIZE, **out},
              open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
