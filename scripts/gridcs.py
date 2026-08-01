"""Grid-based compressive sensing for AFDM channel estimation.

Instead of finding P sharp peaks and refining, treat the channel as sparse
over a fine 2D (delay, Doppler) grid. Solve a sparse linear system, threshold
to keep significant support, then LM-refine on-grid to fractional positions.

Combines with iterative data-aided to close the pilot-bias floor.

Pipeline:
  Fine grid: G_ell x G_kap candidates (say 11 x 21 = 231 = ~10x N_p pilots).
  1. Build atom dictionary D of shape (N, G) at pilot positions.
  2. Solve group-sparse regression for h_grid.
  3. Threshold to top-P_max active grid points.
  4. Data-aided iteration:
     a. LS h on active grid points
     b. Detect symbols
     c. Update pseudo-pilots
     d. Repeat
  5. Optional: LM refine active grid points off-grid
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


def build_grid(N: int, ell_max: float, kappa_max: float, device: str,
               G_ell_per_unit: int = 1, G_kap_per_unit: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ell_grid, kap_grid) each shape (G,) with G = G_ell * G_kap."""
    G_ell = int(ell_max * G_ell_per_unit) + 1
    G_kap = int(2 * kappa_max * G_kap_per_unit) + 1
    ell_1d = torch.linspace(0, ell_max, G_ell, device=device)
    kap_1d = torch.linspace(-kappa_max, kappa_max, G_kap, device=device)
    ell_g, kap_g = torch.meshgrid(ell_1d, kap_1d, indexing="ij")
    return ell_g.reshape(-1), kap_g.reshape(-1)   # (G,), (G,)


def gridcs_receiver(
    system, batch, const, pp, pv, cfg,
    G_ell_per_unit: int = 2, G_kap_per_unit: int = 3,
    P_keep: int = 8,
    n_outer: int = 6,
    n_lm_per_outer: int = 3,
    rho_min: float = 0.9,
    lambda_ridge: float = 1e-2,
):
    r = batch["r"]; y = batch["y"]; sigma_w2 = batch["sigma_w2_block"]
    B, N = r.shape
    device = r.device; dtype = r.dtype

    # 1. Build fine grid.
    ell_g, kap_g = build_grid(N, cfg.ell_max, cfg.kappa_max, device,
                              G_ell_per_unit, G_kap_per_unit)   # (G,)
    G = ell_g.shape[0]
    # Broadcast to (B, G)
    ell_all = ell_g.unsqueeze(0).expand(B, -1).contiguous()
    kap_all = kap_g.unsqueeze(0).expand(B, -1).contiguous()

    # 2. Pilot-only regression on the full grid: solve sparse-like LS.
    x_pilot = torch.zeros(B, N, dtype=dtype, device=device)
    x_pilot[:, pp] = pv.unsqueeze(0)
    A = build_regression_matrix(system, ell_all, kap_all, x_pilot)   # (B, N, G)
    AH = A.conj().transpose(-1, -2)
    AhA = AH @ A                                                     # (B, G, G)
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)                         # (B, G)
    # Ridge is essential since AhA is rank ~ N_p.
    ridge = lambda_ridge * torch.eye(G, dtype=dtype, device=device).unsqueeze(0)
    h_all = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)  # (B, G)

    # 3. Threshold: keep top-P_keep per batch.
    top_val, top_idx = h_all.abs().topk(k=P_keep, dim=-1)               # (B, P_keep)
    ell_hat = torch.gather(ell_all, 1, top_idx)                        # (B, P_keep)
    kap_hat = torch.gather(kap_all, 1, top_idx)
    h_hat = torch.gather(h_all, 1, top_idx)

    # 4. Refit LS on the P_keep active candidates.
    def solve_h(x_ref, ell_c, kap_c):
        A = build_regression_matrix(system, ell_c, kap_c, x_ref)
        AH = A.conj().transpose(-1, -2)
        AhA = AH @ A
        Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
        P = ell_c.shape[1]
        ridge = 1e-4 * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
        return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

    h_hat = solve_h(x_pilot, ell_hat, kap_hat)

    # 5. Data-aided iteration.
    omega = 1.0 / max(sigma_w2, 1e-6)
    def detect(h, ell, kap):
        op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        z = cg_solve(mv, op.rmatvec(y), max_iter=30)
        dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
        p = F.softmax(-omega * dists, dim=-1)
        return p, p.argmax(dim=-1)

    p_ms, hard = detect(h_hat, ell_hat, kap_hat)

    for it in range(n_outer):
        rho = p_ms.max(dim=-1).values
        reliable = rho >= rho_min
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[reliable] = const[hard[reliable]]
        x_hat[:, pp] = pv.unsqueeze(0)

        h_hat = solve_h(x_hat, ell_hat, kap_hat)

        # LM refinement of the P_keep candidates.
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                system, r, h_hat, x_hat, ell_hat, kap_hat,
                sigma_w2=sigma_w2, v_h=None,
                gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
            )

        h_hat = solve_h(x_hat, ell_hat, kap_hat)
        p_ms, hard = detect(h_hat, ell_hat, kap_hat)

    return hard, ell_hat, kap_hat, h_hat


def eval_gridcs(cfg, snr_db, **kwargs):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
    ser_acc = 0.0; n_batches = 8; batch_size = 32
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        hard, _, _, _ = gridcs_receiver(system, batch, const, pp, pv, cfg, **kwargs)
        mask = batch["pilot_mask"]
        ser = float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    for name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    ):
        print()
        print("=" * 78)
        print(f"CONFIG: {name}")
        print("=" * 78)
        for snr in (5.0, 15.0, 25.0):
            ser = eval_gridcs(cfg, snr, G_ell_per_unit=2, G_kap_per_unit=3,
                              P_keep=cfg.P + 2, n_outer=6, n_lm_per_outer=3)
            print(f"  SNR {snr}dB: Grid-CS SER = {ser:.3e}")


if __name__ == "__main__":
    main()
