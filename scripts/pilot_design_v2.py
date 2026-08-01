"""Regenerate Table I (pilot design) with multi-seed error bars.

3 designs (repeated, complementary, hopping) at HARD (P=5, N_p=16), B=4, 15 dB.
Protocol: 3 seeds x 8 batches x 16 realizations per design.
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
from multiblock_dasbl import eval_multiblock


SNR = 15.0
B_BLOCK = 4
DESIGNS = ("repeated", "complementary", "hopping")
N_SEEDS = 3
N_BATCHES = 8
BATCH_SIZE = 16


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                            P=5, N_p=16, P_max=8)
    print(f"\n{'='*70}\nPILOT DESIGN COMPARISON @ HARD (P=5, N_p=16), B={B_BLOCK}, {SNR} dB\n"
          f"K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations\n{'='*70}")
    print(f"{'Design':<15s}  {'mean SER':>12s}  {'std':>10s}")

    results = {}
    for design in DESIGNS:
        vals = []
        t0 = time.time()
        for k in range(N_SEEDS):
            s = eval_multiblock(cfg, SNR, B_block=B_BLOCK, design=design,
                                 n_batches=N_BATCHES, batch_size=BATCH_SIZE,
                                 seed=k * 137 + 42)
            vals.append(s)
        arr = np.array(vals)
        m = float(arr.mean()); std = float(arr.std())
        dt = time.time() - t0
        print(f"{design:<15s}  {m:>12.4e}  {std:>10.4e}   ({dt:.0f}s)")
        results[design] = {"mean": m, "std": std, "seeds": vals}

    out_path = Path("runs/pilot_design_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "B": B_BLOCK,
                   "N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
