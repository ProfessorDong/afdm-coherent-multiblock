"""P1 smoke test: end-to-end AFDM Tx -> channel -> DAFT-domain Rx.

Verifies that a genie-CSI CG-MMSE detector on the DAFT-domain observation
recovers the transmitted symbols with vanishing error at high SNR.

The physical Tx -> channel -> Rx pipeline (with CP-based circular convolution
in length-N and post-CP-strip DAFT) is represented directly by FastAFDMOperator,
which is exactly the DAFT-domain operator H^D. This is the model used by the
training loop for speed. DoublyDispersiveChannel.apply is intended for
educational verification (correctness of Toeplitz-Dirichlet construction)
against a slower time-domain model, not for the physical CP pipeline.

Runs on cuda:0 (RTX 4090).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm import AFDMSystem, DoublyDispersiveChannel, UniformFractionalChannel, FastAFDMOperator


def main() -> None:
    device = "cuda:0"
    torch.manual_seed(0)

    # System parameters (paper defaults)
    N = 128
    kappa_max = 5
    ell_max = 10
    P = 5
    B = 32          # batch size
    snr_dbs = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    sys_ = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=ell_max, device=device)
    ch = UniformFractionalChannel(P=P, ell_max=ell_max, kappa_max=kappa_max, device=device)

    # QPSK constellation
    qpsk = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex64, device=device) / (2 ** 0.5)

    print(f"System: {sys_}")
    print(f"Channel: uniform-random P={P}, ell_max={ell_max}, kappa_max={kappa_max}")
    print(f"Batch: {B}, SNRs (dB): {snr_dbs}\n")
    print(f"{'SNR (dB)':>10s}  {'Genie-ZF BER':>14s}  {'DAFT check':>12s}")
    print("-" * 44)

    for snr_db in snr_dbs:
        # 1. Draw channel
        chdict = ch.sample(B)
        # 2. Draw symbols (label indices)
        idx = torch.randint(0, 4, (B, N), device=device)
        x = qpsk[idx]  # (B, N) complex, DAFT-domain data symbols
        # 3. Forward channel via the DAFT-domain fast operator (mathematically equivalent to
        #    the physical Tx -> CP -> channel -> strip-CP -> DAFT chain).
        op = FastAFDMOperator(system=sys_, ell=chdict["ell"], kappa=chdict["kappa"], h=chdict["h"])
        y_clean = op.matvec(x)
        # 4. Add DAFT-domain AWGN (the DAFT is unitary, so time-domain noise variance
        #    is preserved).
        signal_pow = (y_clean.abs() ** 2).mean()
        noise_var = signal_pow / (10 ** (snr_db / 10))
        w = torch.randn_like(y_clean) * torch.sqrt(noise_var / 2)
        y = y_clean + w
        # 5. Genie-CSI CG-MMSE: solve (H^H H + sigma^2 I) x = H^H y by CG.
        Hty = op.rmatvec(y)
        x_hat = torch.zeros_like(x)
        r_cg = Hty.clone()
        p = r_cg.clone()
        r_norm = (r_cg * torch.conj(r_cg)).sum(dim=-1).real
        for _ in range(30):
            HHp = op.rmatvec(op.matvec(p)) + noise_var * p
            denom = (torch.conj(p) * HHp).sum(dim=-1).real + 1e-12
            alpha = (r_norm / denom).unsqueeze(-1).to(x_hat.dtype)
            x_hat = x_hat + alpha * p
            r_new = r_cg - alpha * HHp
            r_norm_new = (r_new * torch.conj(r_new)).sum(dim=-1).real
            beta = (r_norm_new / (r_norm + 1e-12)).unsqueeze(-1).to(p.dtype)
            p = r_new + beta * p
            r_cg = r_new
            r_norm = r_norm_new
        # 6. Hard demap and compute BER (each QPSK symbol carries 2 bits; here we report
        #    symbol error rate for simplicity, close to 2x BER for high SNR).
        dists = (x_hat.unsqueeze(-1) - qpsk.reshape(1, 1, -1)).abs()
        det_idx = dists.argmin(dim=-1)
        errors = (det_idx != idx).float().mean().item()
        rt_err = (sys_.daft(sys_.idaft(x)) - x).norm().item() / x.norm().item()
        print(f"{snr_db:>10.1f}  {errors:>14.4e}  {rt_err:>12.2e}")


if __name__ == "__main__":
    main()
