"""P2 smoke test: BER-vs-SNR for all three baselines + genie CG-MMSE bound.

Runs on cuda:0 (RTX 4090). Uses uniform-fractional random channels with P=3 paths
so that SBL has a chance to work on the CFAR-initialized candidates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.support import SupportRecovery
from afdm.classical import ClassicalCGDetector, cg_solve
from afdm.pbigabp import PBiGaBPDetector
from afdm.jpnce_sbl import JPNCESBLDetector


def genie_mmse_ser(sys_, chdict, x_true, sigma_w2, qpsk, K_cg=30, pilot_positions=None):
    """Genie-CSI CG-MMSE lower bound."""
    op = FastAFDMOperator(system=sys_, ell=chdict["ell"], kappa=chdict["kappa"], h=chdict["h"])
    y_clean = op.matvec(x_true)
    signal_pow = (y_clean.abs() ** 2).mean()
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    y = y_clean + torch.randn_like(y_clean) * noise_std
    def matvec(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    x_soft = cg_solve(matvec, op.rmatvec(y), max_iter=K_cg)
    dists = (x_soft.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs()
    hard = dists.argmin(dim=-1)
    if pilot_positions is not None:
        mask = torch.ones(sys_.N, dtype=torch.bool, device=y.device); mask[pilot_positions] = False
        true_labels = (x_true.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs().argmin(dim=-1)
        return (hard[:, mask] != true_labels[:, mask]).float().mean().item(), y
    return (hard != (x_true.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs().argmin(dim=-1)).float().mean().item(), y


def main() -> None:
    device = "cuda:0"
    torch.manual_seed(0)
    # System and channel
    N, kappa_max, ell_max, P = 128, 5, 10, 3
    B = 64  # more batches for stable averages
    N_p = 32
    sys_ = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=ell_max, device=device)
    ch = UniformFractionalChannel(P=P, ell_max=ell_max, kappa_max=kappa_max, device=device)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=device, dtype=torch.complex64) / (2 ** 0.5)
    pilot_positions = uniform_daft_pilots(N=N, N_p=N_p, device=device)
    gen = torch.Generator(device=device); gen.manual_seed(0)
    pilot_idx = torch.randint(0, 4, (N_p,), device=device, generator=gen)
    pilot_values = qpsk[pilot_idx]
    mask = torch.ones(N, dtype=torch.bool, device=device); mask[pilot_positions] = False

    sup = SupportRecovery(N=N, N_cp=sys_.ell_max, kappa_max=kappa_max, ell_max=ell_max, P_max=6)

    classical = ClassicalCGDetector(
        system=sys_, support_recovery=sup, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        T=8, K_cg=15, alpha=1.0, lambda_ridge=1e-3,
    )
    pbigabp = PBiGaBPDetector(
        system=sys_, support_recovery=sup, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        T=8, K_cg=15, lambda_h=1e-2, gamma_lr=0.5, gamma_iters=2, omega=20.0, refine_theta=False,
    )
    jpnce = JPNCESBLDetector(
        system=sys_, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        support_recovery=sup,
        T_em=15, T_grid=2, grid_lr=0.05, magnitude_ratio=0.05, K_cg=15,
    )

    snr_dbs = [0, 5, 10, 15, 20, 25]
    print(f"P2 smoke test: N={N}, P={P}, N_p={N_p}, batch={B}")
    print(f"{'SNR (dB)':>8s}  {'Genie':>10s}  {'Classical':>10s}  {'PBiGaBP':>10s}  {'JPNCE-SBL':>10s}  {'time (s)':>9s}")
    print("-" * 72)

    for snr_db in snr_dbs:
        sigma_w2 = 10 ** (-snr_db / 10)
        chdict = ch.sample(B, generator=gen)
        idx = torch.randint(0, 4, (B, N), device=device, generator=gen)
        x = qpsk[idx]; x[:, pilot_positions] = pilot_values.unsqueeze(0)

        # Compute the actual received signal and the ABSOLUTE noise variance
        # (accounting for per-realization signal power, which varies across channels).
        op = FastAFDMOperator(system=sys_, ell=chdict["ell"], kappa=chdict["kappa"], h=chdict["h"])
        y_clean = op.matvec(x)
        signal_pow = (y_clean.abs() ** 2).mean(dim=-1, keepdim=True)  # (B, 1)
        noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
        y = y_clean + torch.randn_like(y_clean) * noise_std
        r = sys_.idaft(y)
        abs_noise_var = (signal_pow.mean() * sigma_w2).item()  # scalar for detector

        t0 = time.time()
        # Genie CG-MMSE
        def matvec_g(v): return op.rmatvec(op.matvec(v)) + abs_noise_var * v
        x_soft = cg_solve(matvec_g, op.rmatvec(y), max_iter=30)
        genie_hard = (x_soft.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs().argmin(dim=-1)
        true_labels = (x.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs().argmin(dim=-1)
        genie_ser = (genie_hard[:, mask] != true_labels[:, mask]).float().mean().item()

        out_c = classical.detect(r, sigma_w2=abs_noise_var)
        ser_c = (out_c["hard_x"][:, mask] != true_labels[:, mask]).float().mean().item()

        out_p = pbigabp.detect(r, sigma_w2=abs_noise_var)
        ser_p = (out_p["hard_x"][:, mask] != true_labels[:, mask]).float().mean().item()

        out_j = jpnce.detect(r, sigma_w2=abs_noise_var)
        ser_j = (out_j["hard_x"][:, mask] != true_labels[:, mask]).float().mean().item()

        dt = time.time() - t0
        print(f"{snr_db:>8d}  {genie_ser:>10.3e}  {ser_c:>10.3e}  {ser_p:>10.3e}  {ser_j:>10.3e}  {dt:>9.2f}")


if __name__ == "__main__":
    main()
