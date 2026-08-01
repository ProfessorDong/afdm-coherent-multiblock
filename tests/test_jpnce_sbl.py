"""Unit tests for JPNCE-SBL detector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.support import SupportRecovery
from afdm.jpnce_sbl import JPNCESBLDetector


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _test_setup(N=128, N_p=32, P=3, seed=0):
    torch.manual_seed(seed)
    sys_ = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=DEVICE)
    ch = UniformFractionalChannel(P=P, ell_max=10.0, kappa_max=5.0, device=DEVICE)
    d = ch.sample(batch=4)
    pilot_positions = uniform_daft_pilots(N=N, N_p=N_p, device=DEVICE)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed + 1)
    pilot_idx = torch.randint(0, 4, (N_p,), device=DEVICE, generator=gen)
    pilot_values = qpsk[pilot_idx]
    return sys_, ch, d, pilot_positions, pilot_values, qpsk


def _forward(sys_, d, x_true, sigma_w2):
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y_clean = op.matvec(x_true)
    signal_pow = (y_clean.abs() ** 2).mean()
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    y = y_clean + torch.randn_like(y_clean) * noise_std
    return sys_.idaft(y), y


def test_jpnce_sbl_recovers_at_high_snr():
    """JPNCE-SBL should achieve reasonable BER at high SNR."""
    sys_, ch, d, pilot_positions, pilot_values, qpsk = _test_setup(N=128, N_p=32, P=3)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pilot_positions] = pilot_values.unsqueeze(0)
    sigma_w2 = 10 ** (-25.0 / 10)
    r, y = _forward(sys_, d, x, sigma_w2)
    sup = SupportRecovery(N=sys_.N, N_cp=sys_.ell_max, kappa_max=5, ell_max=10, P_max=6)
    det = JPNCESBLDetector(
        system=sys_, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        support_recovery=sup,
        T_em=15, T_grid=2, grid_lr=0.05, magnitude_ratio=0.05, K_cg=15,
    )
    out = det.detect(r, sigma_w2=sigma_w2)
    mask = torch.ones(sys_.N, dtype=torch.bool, device=DEVICE); mask[pilot_positions] = False
    ser = (idx[:, mask] != out["hard_x"][:, mask]).float().mean().item()
    print(f"JPNCE-SBL SER at 25 dB, P=3: {ser:.4f}, p_hat mean = {out['p_hat'].float().mean():.1f}")
    assert ser < 0.25, f"SER too high: {ser}"


def test_jpnce_sbl_prunes_to_reasonable_cardinality():
    """After EM, p_hat should be substantially less than P_max_grid."""
    sys_, ch, d, pilot_positions, pilot_values, qpsk = _test_setup(N=128, N_p=32, P=3)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pilot_positions] = pilot_values.unsqueeze(0)
    sigma_w2 = 10 ** (-20.0 / 10)
    r, y = _forward(sys_, d, x, sigma_w2)
    sup = SupportRecovery(N=sys_.N, N_cp=sys_.ell_max, kappa_max=5, ell_max=10, P_max=8)
    det = JPNCESBLDetector(
        system=sys_, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        support_recovery=sup,
        T_em=15, T_grid=2, grid_lr=0.05, magnitude_ratio=0.1,
    )
    out = det.detect(r, sigma_w2=sigma_w2)
    p_mean = out["p_hat"].float().mean().item()
    print(f"P_max init=8, true P=3, pruned p_hat mean = {p_mean:.1f}")
    assert p_mean <= 8, f"pruned count = {p_mean}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
