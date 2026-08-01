"""Verify DAFT/IDAFT unitarity, invertibility, and CPP behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("N,kappa_max,ell_max", [(64, 3, 8), (128, 5, 10), (256, 5, 16)])
def test_daft_unitary(N, kappa_max, ell_max):
    sys = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=ell_max, device=DEVICE)
    # Random complex vector
    x = torch.randn(N, dtype=torch.complex64, device=DEVICE)
    # ||F(x)|| == ||x||
    y = sys.daft(x)
    assert torch.allclose(y.norm(), x.norm(), atol=1e-5), \
        f"DAFT is not unitary: |Fx|={y.norm().item()}, |x|={x.norm().item()}"


@pytest.mark.parametrize("N,kappa_max,ell_max", [(64, 3, 8), (128, 5, 10)])
def test_daft_idaft_inverse(N, kappa_max, ell_max):
    sys = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=ell_max, device=DEVICE)
    x = torch.randn(N, dtype=torch.complex64, device=DEVICE)
    # F^{-1}(F(x)) = x
    x_rec = sys.idaft(sys.daft(x))
    err = (x_rec - x).norm().item()
    assert err < 1e-5, f"IDAFT(DAFT(x)) != x, error = {err}"
    # F(F^{-1}(x)) = x
    x_rec = sys.daft(sys.idaft(x))
    err = (x_rec - x).norm().item()
    assert err < 1e-5, f"DAFT(IDAFT(x)) != x, error = {err}"


def test_batch_daft():
    N = 64
    B = 5
    sys = AFDMSystem(N=N, kappa_max=3, ell_max=8, device=DEVICE)
    X = torch.randn(B, N, dtype=torch.complex64, device=DEVICE)
    Y = sys.daft(X)
    # Compare against per-sample
    for b in range(B):
        y_b = sys.daft(X[b])
        assert torch.allclose(Y[b], y_b, atol=1e-6)


def test_prefix_roundtrip():
    N = 128
    sys = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=DEVICE)
    s = torch.randn(N, dtype=torch.complex64, device=DEVICE)
    s_wcp = sys.add_prefix(s)
    assert s_wcp.shape[-1] == N + sys.ell_max
    s_back = sys.remove_prefix(s_wcp)
    assert torch.allclose(s_back, s)


def test_chirp_params_default():
    N, kappa_max = 128, 5
    sys = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=10, device=DEVICE)
    # c_1 = (2 kappa_max + 1) / (2N) = 11 / 256
    assert abs(sys.c1 - 11.0 / 256) < 1e-9
    # 2 N c_1 should be integer 11
    assert abs(2 * N * sys.c1 - 11) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
