"""Deeper diagnosis: is the plateau caused by failing support recovery?

Runs 4 variants of classical CG at multiple config points:
  (a) N_p=16, P=5   ← publication config
  (b) N_p=32, P=5   ← more pilots
  (c) N_p=16, P=3   ← fewer paths
  (d) N_p=32, P=3   ← easier baseline (matches P2 smoke test)

For each config, tests classical CG with:
  * CFAR-recovered support (real detector)
  * GENIE (true) support (isolates gain-estimation error)

If genie-support classical works everywhere and CFAR-support fails at (a),
we've localized the bug to support recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig
from afdm.channels import UniformFractionalChannel
from afdm.classical import ClassicalCGDetector, build_regression_matrix, cg_solve
from afdm.operators import FastAFDMOperator
from afdm.support import SupportRecovery
from afdm.training import sample_batch


@torch.no_grad()
def genie_support_classical(cfg, snr_db, n_batches=4, batch_size=32, T=8, K_cg=10, seed=42):
    """Classical CG-MMSE using GENIE (true) support instead of CFAR-recovered."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_sum = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        # True support and h are in batch
        ell_true = batch["theta_true"][..., 0]
        kap_true = batch["theta_true"][..., 1]
        # Initialize x_hat: pilots known + zeros
        x_hat = torch.zeros(batch_size, system.N, dtype=torch.complex64, device=cfg.device)
        x_hat[:, pp] = pv.unsqueeze(0)
        r = batch["r"]
        # Alternate T iterations of (ridge LS h) + (CG-MMSE x)
        lam = 1e-3
        eye_p = torch.eye(cfg.P, dtype=torch.complex64, device=cfg.device).unsqueeze(0)
        h_hat = None
        for t in range(T):
            A = build_regression_matrix(system, ell_true, kap_true, x_hat)
            AH = A.conj().transpose(-1, -2)
            M = AH @ A + lam * eye_p
            Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
            h_hat = torch.linalg.solve(M, Ahr.unsqueeze(-1)).squeeze(-1)
            op = FastAFDMOperator(system=system, ell=ell_true, kappa=kap_true, h=h_hat)
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
            # Hard demap + restore pilots
            dists = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs()
            hard = dists.argmin(dim=-1)
            x_hat = const[hard]
            x_hat[:, pp] = pv.unsqueeze(0)
        ser = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
        ser_sum += ser.item()
    return ser_sum / n_batches


@torch.no_grad()
def cfar_support_classical(cfg, snr_db, n_batches=4, batch_size=32, seed=42):
    """Classical CG using CFAR-recovered support (matches ClassicalCGDetector)."""
    pp, pv = cfg.pilots()
    det = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_sum = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        out = det.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
        ser = ((out["hard_x"] != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
        ser_sum += ser.item()
    return ser_sum / n_batches


@torch.no_grad()
def support_recovery_quality(cfg, snr_db, n_batches=4, batch_size=32, seed=42):
    """Measure how accurate CFAR + Newton support recovery is."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    sup = cfg.support_recovery()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    # Reference pilot signal
    x_pilot = torch.zeros(system.N, dtype=torch.complex64, device=cfg.device)
    x_pilot[pp] = pv
    s_pilot = system.idaft(x_pilot.unsqueeze(0))[0]
    total_ell_err = 0.0; total_kap_err = 0.0; total_p_diff = 0
    n_paths = 0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        ell_hat, kappa_hat, p_hat = sup(batch["r"], s_pilot.unsqueeze(0).expand(batch_size, -1))
        # Best-match RMSE (each true path to closest hat)
        for b in range(batch_size):
            for i in range(cfg.P):
                e_true = batch["theta_true"][b, i, 0].item()
                k_true = batch["theta_true"][b, i, 1].item()
                dists = ((ell_hat[b] - e_true) ** 2 + (kappa_hat[b] - k_true) ** 2)
                idx = dists.argmin()
                total_ell_err += (ell_hat[b, idx] - e_true) ** 2
                total_kap_err += (kappa_hat[b, idx] - k_true) ** 2
                n_paths += 1
        total_p_diff += (p_hat - cfg.P).float().abs().sum().item()
    return {
        "delay_rmse": (total_ell_err / n_paths).sqrt().item(),
        "doppler_rmse": (total_kap_err / n_paths).sqrt().item(),
        "avg_p_hat_diff": total_p_diff / (n_batches * batch_size),
    }


def main():
    configs = [
        ("A. N_p=16, P=5 (publication)", 128, 16, 5, 5),
        ("B. N_p=32, P=5",              128, 32, 5, 5),
        ("C. N_p=16, P=3",              128, 16, 3, 3),
        ("D. N_p=32, P=3 (easier)",    128, 32, 3, 3),
    ]
    snrs = [5.0, 15.0, 25.0]
    print("=" * 96)
    print("SUPPORT-RECOVERY DIAGNOSIS")
    print("=" * 96)
    for name, N, N_p, P, P_max in configs:
        cfg = ExperimentConfig(
            N=N, kappa_max=5.0, ell_max=10.0, P=P, N_p=N_p,
            T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=P_max, seed=0,
        )
        print(f"\n{name}")
        print("-" * 96)
        print(f"  {'SNR':>6s}    {'CFAR_class':>12s}  {'Genie_class':>12s}  "
              f"{'delay_RMSE':>12s}  {'kap_RMSE':>12s}  {'P_hat_off':>10s}")
        for snr in snrs:
            cfar_ser = cfar_support_classical(cfg, snr, n_batches=3, batch_size=16)
            genie_ser = genie_support_classical(cfg, snr, n_batches=3, batch_size=16)
            supq = support_recovery_quality(cfg, snr, n_batches=3, batch_size=16)
            print(f"  {snr:>4.0f}dB    {cfar_ser:>12.3e}  {genie_ser:>12.3e}  "
                  f"{supq['delay_rmse']:>12.3f}  {supq['doppler_rmse']:>12.3f}  "
                  f"{supq['avg_p_hat_diff']:>10.2f}")


if __name__ == "__main__":
    main()
