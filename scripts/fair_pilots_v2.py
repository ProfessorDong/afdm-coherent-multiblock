"""Regenerate Table III (fair aggregate pilot) with multi-seed error bars.

For each of two P values {3, 5} and 6 methods (aggregate B*N_p = 64):
  - Classical CG (N_p=64)
  - JPNCE-SBL (N_p=64)
  - SB-IDAR (N_p=64)
  - MB-IDAR B=2, N_p=32 (hopping)
  - MB-IDAR B=4, N_p=16 (hopping)
  - MB-IDAR B=8, N_p=8  (hopping)

Multi-seed: K=3 seeds x 8 batches x 16 realizations = 384 per config.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.classical import ClassicalCGDetector
from afdm.experiments import ExperimentConfig
from afdm.jpnce_sbl import JPNCESBLDetector
from afdm.training import sample_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import receiver_ser
from multiblock_dasbl import eval_multiblock


SNR = 15.0
N_SEEDS = 3
N_BATCHES = 8
BATCH_SIZE = 16


def baseline_ser_seed(det, cfg, snr_db, seed, n_batches=N_BATCHES, batch_size=BATCH_SIZE):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        try:
            out = det.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
            hard = out["hard_x"]
            mask = batch["pilot_mask"]
            ser = float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
        except Exception:
            ser = float("nan")
        ser_acc += ser
    return ser_acc / n_batches


def multi_seed_avg(fn, n_seeds=N_SEEDS):
    vals = [fn(k * 137 + 42) for k in range(n_seeds)]
    return float(np.mean(vals)), float(np.std(vals))


def eval_row(cfg, method):
    """method is one of {'classical', 'jpnce', 'sb_idar', 'mb2', 'mb4', 'mb8'}"""
    system = cfg.system(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    if method == "classical":
        det = ClassicalCGDetector(system=system, support_recovery=cfg.support_recovery(),
                                  constellation=const, pilot_positions=pp, pilot_values=pv,
                                  T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3)
        return multi_seed_avg(lambda seed: baseline_ser_seed(det, cfg, SNR, seed))

    if method == "jpnce":
        det = JPNCESBLDetector(system=system, constellation=const,
                                pilot_positions=pp, pilot_values=pv,
                                support_recovery=cfg.support_recovery(),
                                T_em=15, T_grid=3, grid_lr=0.05, K_cg=15)
        return multi_seed_avg(lambda seed: baseline_ser_seed(det, cfg, SNR, seed))

    if method == "sb_idar":
        return multi_seed_avg(lambda seed: receiver_ser(cfg, SNR, use_reacq=True,
                                                         n_batches=N_BATCHES,
                                                         batch_size=BATCH_SIZE, seed=seed))

    if method.startswith("mb"):
        B = int(method[2:])
        return multi_seed_avg(lambda seed: eval_multiblock(cfg, SNR, B_block=B, design="hopping",
                                                             n_batches=N_BATCHES,
                                                             batch_size=BATCH_SIZE, seed=seed))

    raise ValueError(method)


def main():
    # 12 configs: 2 P values x 6 methods (aggregate BN_p = 64)
    P_VALUES = (3, 5)
    method_configs = [
        ("classical", 1, 64), ("jpnce", 1, 64), ("sb_idar", 1, 64),
        ("mb2", 2, 32), ("mb4", 4, 16), ("mb8", 8, 8),
    ]

    print(f"\n{'='*90}\nFAIR PILOT-BUDGET COMPARISON @ {SNR} dB, aggregate BN_p=64\n"
          f"K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations\n{'='*90}")
    results = {}
    for P in P_VALUES:
        print(f"\n[P={P}]")
        print(f"  {'method':<12s}  {'B':<3s}  {'N_p':<4s}  {'mean SER':>12s}  {'std':>10s}")
        row = {}
        for method, B, N_p in method_configs:
            P_max = max(P + 3, B + 3)
            cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                    P=P, N_p=N_p, P_max=P_max)
            t0 = time.time()
            try:
                m, s = eval_row(cfg, method)
                dt = time.time() - t0
                print(f"  {method:<12s}  {B:<3d}  {N_p:<4d}  {m:>12.4e}  {s:>10.4e}   ({dt:.0f}s)")
                row[method] = {"B": B, "N_p": N_p, "mean": m, "std": s}
            except Exception as e:
                print(f"  {method:<12s}  {B:<3d}  {N_p:<4d}  FAILED: {e}")
                row[method] = {"B": B, "N_p": N_p, "error": str(e)}
        results[f"P={P}"] = row

    out_path = Path("runs/fair_pilots_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "N_seeds": N_SEEDS, "N_batches": N_BATCHES,
                   "batch_size": BATCH_SIZE, "aggregate_pilots": 64,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
