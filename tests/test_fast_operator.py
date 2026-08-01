"""Verify FastAFDMOperator against a slow dense reference and adjoint identity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, FastAFDMOperator, slow_afdm_operator


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("N,P,ell_list,kappa_list", [
    (32, 2, [1.0, 3.0], [0.0, 1.0]),
    (64, 3, [1.0, 2.0, 4.0], [0.0, 1.0, -1.0]),
])
def test_fast_matches_slow_on_integer_paths(N, P, ell_list, kappa_list):
    """When ell_i are integers and kappa_i are integers, fast operator should match slow."""
    torch.manual_seed(0)
    sys = AFDMSystem(N=N, kappa_max=2, ell_max=4, device=DEVICE)
    ell = torch.tensor([ell_list], device=DEVICE)
    kappa = torch.tensor([kappa_list], device=DEVICE)
    h = torch.randn(1, P, dtype=torch.complex64, device=DEVICE)
    x = torch.randn(1, N, dtype=torch.complex64, device=DEVICE)

    op = FastAFDMOperator(system=sys, ell=ell, kappa=kappa, h=h)
    y_fast = op.matvec(x)

    HD = slow_afdm_operator(sys, ell, kappa, h)  # (1, N, N)
    y_slow = (HD @ x.unsqueeze(-1)).squeeze(-1)

    err = (y_fast - y_slow).norm().item() / max(y_slow.norm().item(), 1e-12)
    assert err < 1e-3, f"fast vs slow relative error = {err} at N={N}, P={P}"


def test_fast_matches_slow_on_fractional_paths():
    """Fractional delay and Doppler: fast (FFT-based fractional shift) should agree with slow (periodic-sinc + Toeplitz)."""
    torch.manual_seed(1)
    N, P = 32, 2
    sys = AFDMSystem(N=N, kappa_max=2, ell_max=4, device=DEVICE)
    ell = torch.tensor([[0.7, 2.3]], device=DEVICE)
    kappa = torch.tensor([[0.4, -0.9]], device=DEVICE)
    h = torch.randn(1, P, dtype=torch.complex64, device=DEVICE)
    x = torch.randn(1, N, dtype=torch.complex64, device=DEVICE)

    op = FastAFDMOperator(system=sys, ell=ell, kappa=kappa, h=h)
    y_fast = op.matvec(x)

    HD = slow_afdm_operator(sys, ell, kappa, h)
    y_slow = (HD @ x.unsqueeze(-1)).squeeze(-1)

    err = (y_fast - y_slow).norm().item() / max(y_slow.norm().item(), 1e-12)
    # Fractional models can differ slightly due to conventions; allow 5% tolerance.
    assert err < 0.05, f"fractional-path fast vs slow relative error = {err}"


def test_adjoint_identity():
    """<H^D x, y> = <x, H^{D H} y> for random x, y."""
    torch.manual_seed(2)
    N, P = 64, 4
    sys = AFDMSystem(N=N, kappa_max=3, ell_max=8, device=DEVICE)
    ell = torch.rand(1, P, device=DEVICE) * 8.0
    kappa = (torch.rand(1, P, device=DEVICE) * 2 - 1) * 3.0
    h = torch.randn(1, P, dtype=torch.complex64, device=DEVICE)
    x = torch.randn(1, N, dtype=torch.complex64, device=DEVICE)
    y = torch.randn(1, N, dtype=torch.complex64, device=DEVICE)

    op = FastAFDMOperator(system=sys, ell=ell, kappa=kappa, h=h)
    lhs = (op.matvec(x) * torch.conj(y)).sum()
    rhs = (x * torch.conj(op.rmatvec(y))).sum()
    err = (lhs - rhs).abs().item() / max(lhs.abs().item(), 1e-12)
    assert err < 1e-4, f"adjoint identity violated: relative error = {err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
