"""SER-vs-B scaling at 15 dB for Fig 8 (multi-seed, Table-II-consistent).

Three operating points:
  - Easy (P=3, N_p=32)
  - Hard (P=5, N_p=16)
  - Hard-Np32 (P=5, N_p=32)

Each: B in {1, 2, 4, 8}, protocol K=3 seeds x 8 batches x 16 realizations.
For B=1, uses the same single-block IDAR as multi_seed_error_bars.py (via
receiver_ser + use_reacq=True); for B >= 2, uses eval_multiblock (hopping design).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afdm.experiments import ExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import receiver_ser, oracletheta_dasbl_ser
from multiblock_dasbl import eval_multiblock


SNR = 15.0
BS = [1, 2, 4, 8]
N_SEEDS = 3
N_BATCHES = 8
BATCH_SIZE = 16


def eval_B(cfg, B):
    vals = []
    for k in range(N_SEEDS):
        seed = k * 137 + 42
        if B == 1:
            s = receiver_ser(cfg, SNR, use_reacq=True,
                              n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed)
        else:
            s = eval_multiblock(cfg, SNR, B_block=B, design="hopping",
                                 n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed)
        vals.append(s)
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def eval_oracle_ceiling(cfg):
    vals = []
    for k in range(N_SEEDS):
        seed = k * 137 + 42
        s = oracletheta_dasbl_ser(cfg, SNR, n_batches=N_BATCHES,
                                   batch_size=BATCH_SIZE, seed=seed)
        vals.append(s)
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def main():
    configs = [
        ("Easy (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("Hard (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("Hard-Np32 (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    ]

    print(f"\n{'='*90}\nSER-vs-B at {SNR} dB   [K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations]\n{'='*90}")
    results = {}
    for name, cfg in configs:
        print(f"\n[{name}]")
        print(f"  {'B':<3s}  {'SER':>12s}  {'std':>12s}")
        per_B = {}
        t0 = time.time()
        for B in BS:
            m, s = eval_B(cfg, B)
            per_B[str(B)] = {"mean": m, "std": s}
            print(f"  {B:<3d}  {m:>12.4e}  {s:>12.4e}")
        # Oracle-θ ceiling
        m_o, s_o = eval_oracle_ceiling(cfg)
        per_B["oracle_theta"] = {"mean": m_o, "std": s_o}
        print(f"  oracle-θ ceiling: {m_o:.4e} ± {s_o:.4e}")
        dt = time.time() - t0
        print(f"  wall time: {dt:.0f}s")
        results[name] = per_B

    out_path = Path("runs/scaling_B_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "N_seeds": N_SEEDS, "N_batches": N_BATCHES,
                   "batch_size": BATCH_SIZE, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
