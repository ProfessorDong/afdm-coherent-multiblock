"""Regenerate Fig 9 recovery-regime map with multi-seed averaging.

For each (P, N_p) in {2,3,5,7} x {8,16,24,32,48} at 15 dB, compute:
  - SER of our single-block receiver (SB-IDAR with re-acquisition)
  - SER of Oracle-theta IDAR

Report log10 gap = log10(SER_SB / SER_oracle).

Protocol: K=2 seeds x 6 batches x 16 realizations per cell (192 realizations),
matching the ber_vs_snr_v2 sweep protocol.
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


SNR = 15.0
P_VALUES = (2, 3, 5, 7)
NP_VALUES = (8, 16, 24, 32, 48)
N_SEEDS = 2
N_BATCHES = 6
BATCH_SIZE = 16


def multi_seed(fn, n_seeds=N_SEEDS):
    vals = [fn(k * 137 + 42) for k in range(n_seeds)]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    print(f"\n{'='*90}\nRECOVERY-REGIME MAP at {SNR} dB (K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations)\n{'='*90}")
    print(f"{'P':<3s}  {'N_p':<4s}  {'SER_SB':>12s}  {'SER_oracle':>12s}  {'log10 gap':>10s}")

    grid_sb = np.zeros((len(P_VALUES), len(NP_VALUES)))
    grid_oracle = np.zeros_like(grid_sb)
    grid_gap = np.zeros_like(grid_sb)

    for i, P in enumerate(P_VALUES):
        for j, N_p in enumerate(NP_VALUES):
            cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                   P=P, N_p=N_p, P_max=P + 3)
            t0 = time.time()
            sb, _   = multi_seed(lambda seed: receiver_ser(cfg, SNR, use_reacq=True,
                                                            n_batches=N_BATCHES,
                                                            batch_size=BATCH_SIZE,
                                                            seed=seed))
            orc, _  = multi_seed(lambda seed: oracletheta_dasbl_ser(cfg, SNR,
                                                                     n_batches=N_BATCHES,
                                                                     batch_size=BATCH_SIZE,
                                                                     seed=seed))
            gap = float(np.log10(max(sb, 1e-6)) - np.log10(max(orc, 1e-6)))
            grid_sb[i, j] = sb
            grid_oracle[i, j] = orc
            grid_gap[i, j] = gap
            dt = time.time() - t0
            print(f"{P:<3d}  {N_p:<4d}  {sb:>12.4e}  {orc:>12.4e}  {gap:>10.3f}   ({dt:.0f}s)")

    out_path = Path("runs/phase_diagram_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "snr_db": SNR,
            "P_values": list(P_VALUES),
            "N_p_values": list(NP_VALUES),
            "N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH_SIZE,
            "ser_sb": grid_sb.tolist(),
            "ser_oracle": grid_oracle.tolist(),
            "log_gap": grid_gap.tolist(),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
