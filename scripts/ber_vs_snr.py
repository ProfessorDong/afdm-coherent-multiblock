"""BER vs SNR curves for the paper.

Compares:
  * Genie MMSE (ceiling)
  * Classical CG detector (baseline)
  * Single-block iterative DASBL with reacquisition (our SISO)
  * Multi-block DASBL B=2 and B=4 (our MIMO of blocks)
  * Oracle-theta DASBL (theoretical upper bound achievable with correct theta)

Two configs: EASY (P=3, N_p=32) and HARD (P=5, N_p=16).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import genie_ser, classical_ser, oracletheta_dasbl_ser, receiver_ser
from multiblock_dasbl import eval_multiblock


def main():
    snrs = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    n_batches = 6; batch_size = 16

    for cfg_name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print(f"\n{'=' * 80}\n{cfg_name}\n{'=' * 80}")
        print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'SB-DASBL':>10s}  {'MB B=2':>10s}  {'MB B=4':>10s}  {'Oracle-θ':>10s}")

        curves = {"snrs": snrs, "genie": [], "classical": [], "sb_dasbl": [],
                  "mb_b2": [], "mb_b4": [], "oracle_theta": []}
        for snr in snrs:
            t0 = time.time()
            g = genie_ser(cfg, snr, n_batches=n_batches, batch_size=batch_size)
            c = classical_ser(cfg, snr, n_batches=n_batches, batch_size=batch_size)
            sb = receiver_ser(cfg, snr, use_reacq=True, n_batches=n_batches, batch_size=batch_size)
            mb2 = eval_multiblock(cfg, snr, B_block=2, design="hopping",
                                  n_batches=n_batches, batch_size=batch_size)
            mb4 = eval_multiblock(cfg, snr, B_block=4, design="hopping",
                                  n_batches=n_batches, batch_size=batch_size)
            o = oracletheta_dasbl_ser(cfg, snr, n_batches=n_batches, batch_size=batch_size)
            dt = time.time() - t0
            curves["genie"].append(g); curves["classical"].append(c)
            curves["sb_dasbl"].append(sb); curves["mb_b2"].append(mb2)
            curves["mb_b4"].append(mb4); curves["oracle_theta"].append(o)
            print(f"{snr:>4.1f}dB  {g:>10.3e}  {c:>10.3e}  {sb:>10.3e}  {mb2:>10.3e}  {mb4:>10.3e}  {o:>10.3e}  ({dt:.0f}s)")

        out_path = Path(f"runs/ber_vs_snr_{cfg.P}_{cfg.N_p}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(curves, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
