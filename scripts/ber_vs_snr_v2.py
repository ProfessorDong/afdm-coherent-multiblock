"""SER vs SNR curves for the paper (multi-seed, Table-II-consistent).

Same configurations and receivers as ber_vs_snr.py, but averaged over
K=3 seeds x 8 batches x 32 realizations per (SNR, receiver, config), matching
the multi_seed_error_bars.py protocol at each SNR point.

Reproduces the numbers in Table II at SNR=15 dB and extrapolates to other SNR
values so that Figs 6/7 (SER-vs-SNR) are drawn from the same protocol as the
15-dB comparison table.
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
from phase_diagram import genie_ser, classical_ser, oracletheta_dasbl_ser, receiver_ser
from multiblock_dasbl import eval_multiblock


N_SEEDS = 2
N_BATCHES = 6
BATCH_SIZE = 16


def multi_seed_avg(fn, n_seeds=N_SEEDS):
    vals = [fn(k * 137 + 42) for k in range(n_seeds)]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    snrs = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    for cfg_name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print(f"\n{'=' * 100}\n{cfg_name}   [multi-seed: K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations]\n{'=' * 100}")
        print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'SB-IDAR':>10s}  {'MB B=2':>10s}  {'MB B=4':>10s}  {'Oracle-θ':>10s}")

        curves = {"snrs": snrs, "genie": [], "classical": [], "sb_idar": [],
                  "mb_b2": [], "mb_b4": [], "oracle_theta": [],
                  "genie_std": [], "classical_std": [], "sb_idar_std": [],
                  "mb_b2_std": [], "mb_b4_std": [], "oracle_theta_std": []}
        for snr in snrs:
            t0 = time.time()
            g, g_s   = multi_seed_avg(lambda seed: genie_ser(cfg, snr, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            c, c_s   = multi_seed_avg(lambda seed: classical_ser(cfg, snr, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            sb, sb_s = multi_seed_avg(lambda seed: receiver_ser(cfg, snr, use_reacq=True, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            m2, m2_s = multi_seed_avg(lambda seed: eval_multiblock(cfg, snr, B_block=2, design="hopping", n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            m4, m4_s = multi_seed_avg(lambda seed: eval_multiblock(cfg, snr, B_block=4, design="hopping", n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            o, o_s   = multi_seed_avg(lambda seed: oracletheta_dasbl_ser(cfg, snr, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed))
            dt = time.time() - t0
            for key, val in [("genie", g), ("classical", c), ("sb_idar", sb),
                              ("mb_b2", m2), ("mb_b4", m4), ("oracle_theta", o)]:
                curves[key].append(val)
            for key, val in [("genie_std", g_s), ("classical_std", c_s), ("sb_idar_std", sb_s),
                              ("mb_b2_std", m2_s), ("mb_b4_std", m4_s), ("oracle_theta_std", o_s)]:
                curves[key].append(val)
            print(f"{snr:>4.1f}dB  {g:>10.3e}  {c:>10.3e}  {sb:>10.3e}  {m2:>10.3e}  {m4:>10.3e}  {o:>10.3e}   ({dt:.0f}s)")

        out_path = Path(f"runs/ber_vs_snr_v2_{cfg.P}_{cfg.N_p}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(curves, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
