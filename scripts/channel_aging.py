"""Channel-aging robustness sweep for the shared-support assumption.

Theorem 2's 1/B Fisher-info gain assumes theta is exactly constant across the
B-block coherence window. This script tests how quickly performance degrades
when theta drifts across blocks: for each block b we perturb the true theta by
delta_b ~ Uniform(-drift, drift) *per component* (delay in samples,
Doppler in normalized units), then simulate the received signal with the
per-block operator. The receiver still assumes shared theta (uncorrected).

We sweep drift in {0, 0.05, 0.1, 0.2, 0.3, 0.5} and report SER at 15 dB for
B in {1, 4, 8} on the HARD config.
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
from afdm.multi_block import (
    MultiBlockBatch,
    PILOT_DESIGNS,
    block_doppler_phase,
    sample_multiblock,
)
from afdm.operators import FastAFDMOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver


N_SEEDS = 5
N_BATCHES = 4
BATCH_SIZE = 16
SNR = 15.0
DRIFTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
BS = [1, 4, 8]


def sample_aged_batch(system, channel, const, pp, pv, batch_size, snr_db,
                      drift, generator):
    """Same as sample_multiblock but injects per-block theta drift.

    drift: half-width of Uniform per-component perturbation. Applied to
    both ell (delay in sample units) and kap (normalized Doppler index).
    """
    device = system.device; N = system.N; B_block = pp.shape[0]
    S = const.numel()

    d = channel.sample(batch_size, generator=generator)
    h_true = d["h"]; ell0 = d["ell"]; kap0 = d["kappa"]
    P = ell0.shape[1]

    # Data symbols per block, then insert pilots.
    idx = torch.randint(0, S, (batch_size, B_block, N), device=device,
                        generator=generator)
    x = const[idx]
    for b in range(B_block):
        x[:, b, pp[b]] = pv[b].unsqueeze(0)
    labels = (x.unsqueeze(-1) - const.reshape(1, 1, 1, -1)).abs().argmin(dim=-1)

    # Per-block drift on theta + block-dependent Doppler phase h_b = h * D_b(kap_b).
    N_cp = system.ell_max
    y_clean_list = []
    for b in range(B_block):
        u_e = (2.0 * torch.rand(batch_size, P, device=device, generator=generator) - 1.0)
        u_k = (2.0 * torch.rand(batch_size, P, device=device, generator=generator) - 1.0)
        ell_b = (ell0 + drift * u_e).clamp(min=0.0, max=system.ell_max)
        kap_b = (kap0 + drift * u_k).clamp(min=-system.kappa_max, max=system.kappa_max)
        phase_b = block_doppler_phase(kap_b, b, N, N_cp)
        h_b = h_true * phase_b
        op_b = FastAFDMOperator(system=system, ell=ell_b, kappa=kap_b, h=h_b)
        y_clean_list.append(op_b.matvec(x[:, b, :]))
    y_clean = torch.stack(y_clean_list, dim=1)

    signal_pow = (y_clean.abs() ** 2).mean()
    sigma_w2 = 10 ** (-snr_db / 10)
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    w = torch.randn(y_clean.shape, dtype=y_clean.dtype, device=device,
                    generator=generator) * noise_std
    y = y_clean + w
    r = system.idaft(y.reshape(-1, N)).reshape(batch_size, B_block, N)

    pilot_mask = torch.ones(batch_size, B_block, N, dtype=torch.bool, device=device)
    for b in range(B_block):
        pilot_mask[:, b, pp[b]] = False

    abs_noise = (signal_pow * sigma_w2).item()
    theta_true = torch.stack([ell0, kap0], dim=-1)
    return MultiBlockBatch(
        r=r, y=y, x_true=x, labels=labels, h_true=h_true, theta_true=theta_true,
        pilot_positions=pp, pilot_values=pv,
        pilot_mask=pilot_mask, sigma_w2_block=abs_noise, snr_db=snr_db,
    )


def eval_aging(cfg, snr_db, B_block, drift, seed):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                       constellation=const, device=cfg.device,
                                       seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(N_BATCHES):
        batch = sample_aged_batch(system, channel, const, pp, pv,
                                  batch_size=BATCH_SIZE, snr_db=snr_db,
                                  drift=drift, generator=gen)
        hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                   n_outer=6, n_lm_per_outer=3,
                                                   rho_min=0.5, use_reacq=True)
        mask = batch.pilot_mask
        ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / N_BATCHES


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    print(f"\n{'='*70}\nCHANNEL-AGING ROBUSTNESS @ 15 dB (HARD, K={N_SEEDS} seeds)\n{'='*70}")

    results = {}
    for B in BS:
        print(f"\nB = {B}")
        print(f"  {'drift':<10s} mean SER   +/- std")
        for drift in DRIFTS:
            t0 = time.time()
            sers = []
            for k in range(N_SEEDS):
                s = eval_aging(cfg, SNR, B_block=B, drift=drift,
                               seed=k * 137 + 42)
                sers.append(s)
            arr = np.array(sers)
            mean = float(arr.mean()); std = float(arr.std())
            results[(B, drift)] = {"mean": mean, "std": std, "seeds": sers}
            dt = time.time() - t0
            print(f"  {drift:<10.3f} {mean:.3e} +/- {std:.3e} ({dt:.0f}s)")

    out = {f"B{b}_d{d}": v for (b, d), v in results.items()}
    out_path = Path("runs/channel_aging.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"BS": BS, "DRIFTS": DRIFTS, "results": out}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
