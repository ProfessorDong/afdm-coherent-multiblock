"""Sensitivity of iterative DASBL to theta initialization.

Question: is the CFAR-init failure of iterative DASBL due to
  (a) The initial theta is outside the LM basin of attraction (fixable if we
      could initialize a bit closer), OR
  (b) A fundamental identifiability limit (multiple theta configs look the
      same under available pilot observations)?

Test: take TRUE theta and perturb by controlled Gaussian noise with sigma =
  {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
in both delay and Doppler. Run iterative DASBL with 6 outer + 3 LM per outer.
Report final SER as a function of sigma.

Interpretation:
  * If SER is roughly constant for small sigma then jumps at some critical
    value -> basin of attraction, deterministic recovery threshold.
  * If SER degrades smoothly -> continuous accuracy dependence.
  * Compare sigma at which recovery breaks against typical CFAR RMSE
    (~0.5 samples for delay, ~0.3 for Doppler).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step


def run(cfg, snr_db, perturb_sigma, n_iters=6, n_lm=3, rho_min=0.5,
        n_batches=8, batch_size=32, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        B, N = batch["r"].shape
        dtype = batch["r"].dtype; device = batch["r"].device
        # Perturb TRUE theta
        ell_true = batch["theta_true"][..., 0]; kap_true = batch["theta_true"][..., 1]
        if perturb_sigma > 0:
            perturb_gen = torch.Generator(device=device)
            perturb_gen.manual_seed(seed + int(perturb_sigma * 1000))
            ell_p = torch.randn(ell_true.shape, dtype=torch.float32, device=device, generator=perturb_gen) * perturb_sigma
            kap_p = torch.randn(kap_true.shape, dtype=torch.float32, device=device, generator=perturb_gen) * perturb_sigma
            ell_hat = (ell_true + ell_p).clamp(min=0, max=cfg.ell_max)
            kap_hat = (kap_true + kap_p).clamp(min=-cfg.kappa_max, max=cfg.kappa_max)
        else:
            ell_hat = ell_true.clone()
            kap_hat = kap_true.clone()

        # Initial pilot-only LS
        def solve_h(x_ref, ell_c, kap_c):
            A = build_regression_matrix(system, ell_c, kap_c, x_ref)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell_c.shape[1]
            ridge = 1e-3 * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
            return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, pp] = pv.unsqueeze(0)
        h_hat = solve_h(x_hat, ell_hat, kap_hat)

        omega = 1.0 / max(batch["sigma_w2_block"], 1e-6)
        def detect(h, ell, kap):
            op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p = F.softmax(-omega * dists, dim=-1)
            return p, p.argmax(dim=-1)

        p_ms, hard = detect(h_hat, ell_hat, kap_hat)

        for it in range(n_iters):
            rho = p_ms.max(dim=-1).values
            reliable = rho >= rho_min
            x_hat_it = torch.zeros(B, N, dtype=dtype, device=device)
            x_hat_it[reliable] = const[hard[reliable]]
            x_hat_it[:, pp] = pv.unsqueeze(0)
            h_hat = solve_h(x_hat_it, ell_hat, kap_hat)
            for _ in range(n_lm):
                ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                    system, batch["r"], h_hat, x_hat_it, ell_hat, kap_hat,
                    sigma_w2=batch["sigma_w2_block"], v_h=None,
                    gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
                )
            h_hat = solve_h(x_hat_it, ell_hat, kap_hat)
            p_ms, hard = detect(h_hat, ell_hat, kap_hat)

        mask = batch["pilot_mask"]
        ser = float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    sigmas = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
    for name, cfg in (
        ("HARD (P=5, N_p=16) @ 15dB",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("EASY (P=3, N_p=32) @ 15dB",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
    ):
        print()
        print("=" * 78)
        print(f"CONFIG: {name}")
        print("=" * 78)
        print(f"{'perturb sigma':<14s}  {'final SER':>12s}")
        for s in sigmas:
            ser = run(cfg, snr_db=15.0, perturb_sigma=s)
            print(f"{s:<14.2f}  {ser:>12.3e}")


if __name__ == "__main__":
    main()
