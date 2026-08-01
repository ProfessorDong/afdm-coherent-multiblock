"""Comprehensive baseline comparison for the paper.

Compares:
  * Genie MMSE
  * Classical CG (Xu 2024 style)
  * JPNCE-SBL (Xu 2026)
  * PBiGaBP (Rasangan 2024)
  * Our single-block iterative DASBL with reacquisition
  * Our multi-block DASBL B=4 and B=8
  * Oracle-theta DASBL (data-aided upper bound)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.classical import ClassicalCGDetector
from afdm.experiments import ExperimentConfig
from afdm.jpnce_sbl import JPNCESBLDetector
from afdm.pbigabp import PBiGaBPDetector
from afdm.training import sample_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import genie_ser, oracletheta_dasbl_ser, receiver_ser
from multiblock_dasbl import eval_multiblock


def baseline_ser(det, cfg, snr_db, n_batches, batch_size, seed=42):
    """Generic baseline evaluator using .detect() interface."""
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
        except Exception as e:
            print(f"    {type(det).__name__} failed: {type(e).__name__}: {str(e)[:60]}")
            ser = float("nan")
        ser_acc += ser
    return ser_acc / n_batches


def main():
    snrs = [5.0, 15.0, 25.0]
    n_batches = 4; batch_size = 16

    for cfg_name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print("\n" + "=" * 100)
        print(f"CONFIG: {cfg_name}")
        print("=" * 100)

        system = cfg.system(); const = cfg.constellation()
        pp, pv = cfg.pilots()
        support = cfg.support_recovery()
        classical = ClassicalCGDetector(system=system, support_recovery=support,
                                        constellation=const, pilot_positions=pp,
                                        pilot_values=pv, T=8, K_cg=10,
                                        alpha=1.0, lambda_ridge=1e-3)
        jpnce = JPNCESBLDetector(system=system, constellation=const,
                                 pilot_positions=pp, pilot_values=pv,
                                 support_recovery=support, T_em=15, T_grid=3,
                                 grid_lr=0.05, K_cg=15)
        try:
            pbigabp = PBiGaBPDetector(system=system, support_recovery=support,
                                       constellation=const, pilot_positions=pp,
                                       pilot_values=pv)
        except Exception as e:
            print(f"  PBiGaBP init failed: {e}")
            pbigabp = None

        print(f"{'method':<25s}  " + "  ".join(f"{snr:>5.1f}dB" for snr in snrs))
        for snr in snrs:
            print()
            print(f"  SNR = {snr}")
            g = genie_ser(cfg, snr, n_batches=n_batches, batch_size=batch_size)
            print(f"    genie MMSE           = {g:.3e}")
            c = baseline_ser(classical, cfg, snr, n_batches, batch_size)
            print(f"    classical CG         = {c:.3e}")
            j = baseline_ser(jpnce, cfg, snr, n_batches, batch_size)
            print(f"    JPNCE-SBL            = {j:.3e}")
            if pbigabp is not None:
                p = baseline_ser(pbigabp, cfg, snr, n_batches, batch_size)
                print(f"    PBiGaBP              = {p:.3e}")
            sb = receiver_ser(cfg, snr, use_reacq=True, n_batches=n_batches,
                              batch_size=batch_size)
            print(f"    SB-DASBL (ours)      = {sb:.3e}")
            mb4 = eval_multiblock(cfg, snr, B_block=4, design="hopping",
                                  n_batches=n_batches, batch_size=batch_size)
            print(f"    MB-DASBL B=4 (ours)  = {mb4:.3e}")
            mb8 = eval_multiblock(cfg, snr, B_block=8, design="hopping",
                                  n_batches=n_batches, batch_size=batch_size)
            print(f"    MB-DASBL B=8 (ours)  = {mb8:.3e}")
            o = oracletheta_dasbl_ser(cfg, snr, n_batches=n_batches, batch_size=batch_size)
            print(f"    oracle-theta DASBL   = {o:.3e}")


if __name__ == "__main__":
    main()
