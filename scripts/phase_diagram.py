"""Identifiability phase diagram for single-block off-grid AFDM.

Sweep (P, N_p, SNR) and characterize:
  * Achievable SER by iterative DASBL with data-aided re-acquisition (our receiver)
  * Genie MMSE ceiling (perfect channel knowledge)
  * Iterative DASBL with oracle theta (data-aided ceiling, isolates theta cost)

Region I:   SER_receiver ~= SER_genie  (identifiable, receiver at CRB)
Region II:  SER_receiver > SER_genie but << classical (partial recovery)
Region III: SER_receiver ~= classical (ambiguity-limited, no algorithmic fix)

The phase diagram is the paper's central artifact: it tells engineers whether
a given (P, N_p) design point can support high-quality reception with our
algorithm, or whether they need to change the design (more pilots, multi-block).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import ClassicalCGDetector, build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_dasbl_reacq import dasbl_reacq


def genie_ser(cfg, snr_db, n_batches=6, batch_size=32, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv, batch_size=batch_size,
                             snr_db=snr_db, generator=gen)
        op = FastAFDMOperator(system=system, ell=batch["theta_true"][..., 0],
                              kappa=batch["theta_true"][..., 1], h=batch["h_true"])
        def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
        x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
        hard = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def classical_ser(cfg, snr_db, n_batches=6, batch_size=32, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    det = ClassicalCGDetector(system=system, support_recovery=cfg.support_recovery(),
                              constellation=const, pilot_positions=pp, pilot_values=pv,
                              T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv, batch_size=batch_size,
                             snr_db=snr_db, generator=gen)
        out = det.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
        ser = float(((out["hard_x"] != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def receiver_ser(cfg, snr_db, use_reacq=True, n_batches=6, batch_size=32, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv, batch_size=batch_size,
                             snr_db=snr_db, generator=gen)
        hard, _, _, _ = dasbl_reacq(system, batch, const, pp, pv, cfg,
                                    n_outer=8, n_lm_per_outer=2, rho_min=0.9,
                                    use_reacq=use_reacq)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def oracletheta_dasbl_ser(cfg, snr_db, n_batches=6, batch_size=32, seed=42):
    """Iterative DASBL with TRUE theta — the fundamental data-aided ceiling."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv, batch_size=batch_size,
                             snr_db=snr_db, generator=gen)
        ell = batch["theta_true"][..., 0]; kap = batch["theta_true"][..., 1]
        B, N = batch["r"].shape
        dtype = batch["r"].dtype; device = batch["r"].device

        def solve_h(x_ref):
            A = build_regression_matrix(system, ell, kap, x_ref)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell.shape[1]
            ridge = 1e-3 * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
            return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

        x_hat = torch.zeros(B, N, dtype=dtype, device=device); x_hat[:, pp] = pv.unsqueeze(0)
        h = solve_h(x_hat)
        omega = 1.0 / max(batch["sigma_w2_block"], 1e-6)
        for _ in range(5):
            op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms = F.softmax(-omega * dists, dim=-1)
            hard = p_ms.argmax(dim=-1)
            rho = p_ms.max(dim=-1).values
            reliable = rho >= 0.9
            x_hat_it = torch.zeros(B, N, dtype=dtype, device=device)
            x_hat_it[reliable] = const[hard[reliable]]
            x_hat_it[:, pp] = pv.unsqueeze(0)
            h = solve_h(x_hat_it)
        # Final detect
        op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
        def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
        z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
        hard = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def make_cfg(P, N_p):
    return ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=P, N_p=N_p,
                            P_max=P + 3)


def main():
    P_values = (2, 3, 5, 7)
    Np_values = (8, 16, 24, 32, 48)
    SNR = 15.0

    print("=" * 100)
    print(f"IDENTIFIABILITY PHASE DIAGRAM at SNR = {SNR} dB, N = 128, kappa_max = 5")
    print("=" * 100)
    print(f"\nEach cell: 'genie / classical / receiver / oracle-theta DASBL' SER")
    print(f"{'':>4s}  " + "  ".join(f"{'N_p=%d' % n:>28s}" for n in Np_values))
    for P in P_values:
        print(f"P={P:<2d}  ", end="")
        for N_p in Np_values:
            if N_p >= 8:  # some cfgs may not have enough pilots
                cfg = make_cfg(P, N_p)
                t0 = time.time()
                g = genie_ser(cfg, SNR)
                c = classical_ser(cfg, SNR)
                rx = receiver_ser(cfg, SNR, use_reacq=True)
                o = oracletheta_dasbl_ser(cfg, SNR)
                dt = time.time() - t0
                print(f"{g:.2e}/{c:.2e}/{rx:.2e}/{o:.2e} ({dt:.0f}s)", end="  ")
        print()

    print("\n" + "=" * 100)
    print("KEY: g = genie MMSE, c = classical CG, rx = our receiver, o = oracle-theta DASBL")
    print("Region I (rx ~ g):  identifiable, our receiver is near-optimal")
    print("Region II (rx << c but rx > g): partial recovery, room for improvement")
    print("Region III (rx ~ c): ambiguity-limited, need multi-block or fewer paths")


if __name__ == "__main__":
    main()
