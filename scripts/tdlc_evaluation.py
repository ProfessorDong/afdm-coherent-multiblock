"""TDL-C channel model evaluation.

Standardized 3GPP TS 38.901 tapped delay line profile. Tests generalization of
our multi-block DASBL receiver from the synthetic Uniform channel to a
realistic 3GPP model.

Key parameters:
  * N = 128, delta_f = 15 kHz (5G NR-like)
  * TDL-C delay profile with P_use = 5 or 7 dominant taps
  * doppler_hz = 500 Hz -> kappa = doppler/delta_f = 0.033 (very low)
    or  doppler_hz = 5000 Hz -> kappa = 0.33 (moderate)
  * delay_spread_ns = 100-300 ns -> ell varies

We compare against classical CG and JPNCE-SBL baselines.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.channels import TDLProfile
from afdm.classical import ClassicalCGDetector, cg_solve
from afdm.experiments import ExperimentConfig, qpsk_constellation
from afdm.jpnce_sbl import JPNCESBLDetector
from afdm.multi_block import (
    PILOT_DESIGNS, sample_multiblock, MultiBlockBatch, block_doppler_phase,
)
from afdm.operators import FastAFDMOperator
from afdm.support import SupportRecovery
from afdm.system import AFDMSystem
from afdm.training import sample_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver


def sample_multiblock_tdl(system, tdl_channel, constellation, pp, pv,
                          batch_size, snr_db, generator, N):
    """Multi-block batch sampling using TDLProfile (shared theta across blocks)."""
    device = system.device
    N_block = pp.shape[0]
    # Sample TDL channel realizations (shared per batch element across blocks)
    d = tdl_channel.sample(batch_size, N, generator=generator)
    h_true = d["h"]; ell = d["ell"]; kap = d["kappa"]
    S = constellation.numel()

    if generator is None:
        idx = torch.randint(0, S, (batch_size, N_block, N), device=device)
    else:
        idx = torch.randint(0, S, (batch_size, N_block, N), device=device, generator=generator)
    x = constellation[idx]
    for b in range(N_block):
        x[:, b, pp[b]] = pv[b].unsqueeze(0)
    labels = (x.unsqueeze(-1) - constellation.reshape(1, 1, 1, -1)).abs().argmin(dim=-1)

    # Apply per-block Doppler phase h_b = h_true * D_b(kappa)
    N_cp = system.ell_max
    y_clean_list = []
    for b in range(N_block):
        phase_b = block_doppler_phase(kap, b, N, N_cp)
        h_b = h_true * phase_b
        op_b = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h_b)
        y_clean_list.append(op_b.matvec(x[:, b, :]))
    y_clean = torch.stack(y_clean_list, dim=1)
    signal_pow = (y_clean.abs() ** 2).mean()
    sigma_w2 = 10 ** (-snr_db / 10)
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    if generator is None:
        w = torch.randn_like(y_clean) * noise_std
    else:
        w = torch.randn(y_clean.shape, dtype=y_clean.dtype, device=device, generator=generator) * noise_std
    y = y_clean + w
    r = system.idaft(y.reshape(-1, N)).reshape(batch_size, N_block, N)

    pilot_mask = torch.ones(batch_size, N_block, N, dtype=torch.bool, device=device)
    for b in range(N_block):
        pilot_mask[:, b, pp[b]] = False

    abs_noise = (signal_pow * sigma_w2).item()
    theta_true = torch.stack([ell, kap], dim=-1)
    return MultiBlockBatch(
        r=r, y=y, x_true=x, labels=labels, h_true=h_true, theta_true=theta_true,
        pilot_positions=pp, pilot_values=pv, pilot_mask=pilot_mask,
        sigma_w2_block=abs_noise, snr_db=snr_db,
    )


def make_tdl_channel(P_use, delay_spread_ns=100, doppler_hz=3000, device="cuda:0"):
    """P_use = number of TDL-C taps to use (top-P_use by power). Delay spread scaled
    so ell values map to reasonable AFDM index range for N=128, delta_f=15 kHz.
    """
    ch = TDLProfile(profile="TDL-C", delay_spread_ns=delay_spread_ns,
                    delta_f_hz=15e3, doppler_hz=doppler_hz, P_use=P_use, device=device)
    return ch


class MockCfg:
    """Config object compatible with multiblock_dasbl_receiver."""
    def __init__(self, system, N_p, P, kappa_max, ell_max, device):
        self.N = system.N
        self.N_p = N_p
        self.P = P
        self.P_max = P + 3
        self.kappa_max = kappa_max
        self.ell_max = ell_max
        self.device = device
    def system(self):
        return _cached_system[0]


def evaluate_tdl(P_use, delay_spread_ns, doppler_hz, N_p, B_block,
                 snr_dbs=(5.0, 15.0, 25.0), n_batches=6, batch_size=16):
    device = "cuda:0"
    N = 128
    kappa_max = 5.0
    ell_max = 10.0
    system = AFDMSystem(N=N, kappa_max=int(kappa_max), ell_max=int(ell_max), device=device)
    _cached_system[0] = system

    tdl = make_tdl_channel(P_use=P_use, delay_spread_ns=delay_spread_ns,
                           doppler_hz=doppler_hz, device=device)
    const = qpsk_constellation(device)
    # Multi-block pilots with hopping design
    pp, pv = PILOT_DESIGNS["hopping"](N=N, N_p=N_p, B=B_block,
                                       constellation=const, device=device, seed=42)
    cfg = MockCfg(system, N_p, P_use, kappa_max, ell_max, device)

    print(f"  TDL-C P_use={P_use}, tau_rms={delay_spread_ns}ns, "
          f"nu_max={doppler_hz}Hz -> kappa_max_actual={doppler_hz/15e3:.3f}, "
          f"N_p={N_p}, B={B_block}")
    results = {}
    for snr in snr_dbs:
        gen = torch.Generator(device=device); gen.manual_seed(42)
        ser_acc = 0.0
        for _ in range(n_batches):
            batch = sample_multiblock_tdl(system, tdl, const, pp, pv,
                                          batch_size=batch_size, snr_db=snr,
                                          generator=gen, N=N)
            with torch.no_grad():
                hard, _, _, _ = multiblock_dasbl_receiver(
                    system, batch, const, cfg,
                    n_outer=6, n_lm_per_outer=3, rho_min=0.5, use_reacq=True,
                )
            mask = batch.pilot_mask
            ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
            ser_acc += ser
        results[snr] = ser_acc / n_batches
        print(f"    SNR {snr}dB: MB-DASBL B={B_block} SER = {results[snr]:.3e}")
    return results


_cached_system = [None]


def main():
    import json
    from pathlib import Path
    print("=" * 80)
    print("TDL-C CHANNEL EVALUATION (3GPP TS 38.901) -- physical model with D_b(kappa)")
    print("=" * 80)

    results = {}
    for P_use in (5, 7):
        for doppler_hz in (500, 3000):
            key = f"P{P_use}_nu{doppler_hz}"
            results[key] = {}
            print(f"\n--- P_use={P_use}, doppler_hz={doppler_hz} ---")
            for B in (1, 2, 4, 8):
                N_p_per = 32 // max(B // 2, 1)   # keeps aggregate ~64
                if N_p_per < 8:
                    N_p_per = 8
                r = evaluate_tdl(P_use=P_use, delay_spread_ns=100, doppler_hz=doppler_hz,
                                 N_p=N_p_per, B_block=B, snr_dbs=(15.0,),
                                 n_batches=8, batch_size=32)
                results[key][str(B)] = {"ser": r[15.0], "N_p": N_p_per}

    out = Path("runs/tdlc_v2.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
