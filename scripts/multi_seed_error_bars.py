"""Multi-seed run for statistical error bars on the headline result.

For each of Easy/Hard configs and each of {classical, JPNCE-SBL, PBiGaBP,
SB-DASBL, MB-DASBL B=4, MB-DASBL B=8, oracle-theta DASBL, genie}, run K=10
independent seeds with 8 batches x 32 realizations each. Report mean +/-
standard deviation of SER.
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
from afdm.pbigabp import PBiGaBPDetector
from afdm.training import sample_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import genie_ser, oracletheta_dasbl_ser, receiver_ser
from multiblock_dasbl import eval_multiblock


N_SEEDS = 10
N_BATCHES = 8
BATCH_SIZE = 32
SNR = 15.0


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
        except Exception as e:
            ser = float("nan")
        ser_acc += ser
    return ser_acc / n_batches


def genie_seed(cfg, snr_db, seed):
    return genie_ser(cfg, snr_db, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed)


def oracle_theta_seed(cfg, snr_db, seed):
    return oracletheta_dasbl_ser(cfg, snr_db, n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed)


def sb_dasbl_seed(cfg, snr_db, seed):
    return receiver_ser(cfg, snr_db, use_reacq=True, n_batches=N_BATCHES,
                         batch_size=BATCH_SIZE, seed=seed)


def mb_dasbl_seed(cfg, snr_db, B, seed):
    return eval_multiblock(cfg, snr_db, B_block=B, design="hopping",
                             n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed=seed)


def main():
    all_results = {}
    for cfg_name, cfg in (
        ("easy", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("hard", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print(f"\n{'='*70}\n{cfg_name.upper()} @ 15 dB (K={N_SEEDS} seeds, 8 batches, 32 realizations)\n{'='*70}")
        cfg_results = {}
        system = cfg.system(); const = cfg.constellation()
        pp, pv = cfg.pilots()
        classical = ClassicalCGDetector(system=system, support_recovery=cfg.support_recovery(),
                                        constellation=const, pilot_positions=pp, pilot_values=pv,
                                        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3)
        jpnce = JPNCESBLDetector(system=system, constellation=const,
                                 pilot_positions=pp, pilot_values=pv,
                                 support_recovery=cfg.support_recovery(),
                                 T_em=15, T_grid=3, grid_lr=0.05, K_cg=15)
        try:
            pbigabp = PBiGaBPDetector(system=system, support_recovery=cfg.support_recovery(),
                                       constellation=const, pilot_positions=pp, pilot_values=pv)
        except Exception:
            pbigabp = None

        methods = [
            ("genie",     lambda seed: genie_seed(cfg, SNR, seed)),
            ("classical", lambda seed: baseline_ser_seed(classical, cfg, SNR, seed)),
            ("jpnce",     lambda seed: baseline_ser_seed(jpnce, cfg, SNR, seed)),
        ]
        if pbigabp is not None:
            methods.append(("pbigabp", lambda seed: baseline_ser_seed(pbigabp, cfg, SNR, seed)))
        methods.extend([
            ("sb_dasbl", lambda seed: sb_dasbl_seed(cfg, SNR, seed)),
            ("mb_b4",    lambda seed: mb_dasbl_seed(cfg, SNR, 4, seed)),
            ("mb_b8",    lambda seed: mb_dasbl_seed(cfg, SNR, 8, seed)),
            ("oracle",   lambda seed: oracle_theta_seed(cfg, SNR, seed)),
        ])

        for name, fn in methods:
            t0 = time.time()
            seed_sers = []
            for seed in range(N_SEEDS):
                s = fn(seed * 137 + 42)  # spread seeds
                seed_sers.append(s)
            arr = np.array(seed_sers)
            mean = arr.mean(); std = arr.std()
            cfg_results[name] = {"seeds": seed_sers, "mean": float(mean), "std": float(std)}
            dt = time.time() - t0
            print(f"  {name:<12s}: {mean:.3e} +/- {std:.3e} ({dt:.0f}s)")
        all_results[cfg_name] = cfg_results

    out_path = Path("runs/multiseed_15db.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
