"""Verify periodic-sinc kernel properties and channel-generation shapes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm.channels import (
    DoublyDispersiveChannel,
    UniformFractionalChannel,
    TDLProfile,
)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_periodic_sinc_zero_delta_is_delta():
    """At delta = 0, the (complex) Dirichlet kernel should be delta[m]."""
    N = 64
    m = torch.arange(N, device=DEVICE, dtype=torch.float32)
    delta = torch.tensor([0.0], device=DEVICE)
    g = DoublyDispersiveChannel.periodic_sinc(m, delta, N).squeeze(0)
    assert torch.allclose(g[0].real, torch.tensor(1.0, device=DEVICE), atol=1e-5)
    assert torch.allclose(g[0].imag, torch.tensor(0.0, device=DEVICE), atol=1e-5)
    assert g[1:].abs().max() < 1e-5


def test_periodic_sinc_energy_conservation():
    """The Dirichlet kernel is an orthonormal fractional-shift filter: unit energy."""
    N = 64
    m = torch.arange(N, device=DEVICE, dtype=torch.float32)
    for delta_val in [0.0, 0.3, 0.5, 0.7, 1.0, 2.5]:
        delta = torch.tensor([delta_val], device=DEVICE)
        g = DoublyDispersiveChannel.periodic_sinc(m, delta, N).squeeze(0)
        energy = (g.abs() ** 2).sum().item()
        assert abs(energy - 1.0) < 1e-4, f"delta={delta_val}: energy={energy}, expected 1.0"


def test_uniform_channel_shape_and_range():
    ch = UniformFractionalChannel(P=5, ell_max=10.0, kappa_max=5.0, device=DEVICE)
    batch = 4096  # large enough to reduce sample variance
    d = ch.sample(batch)
    assert d["ell"].shape == (batch, 5)
    assert d["kappa"].shape == (batch, 5)
    assert d["h"].shape == (batch, 5)
    assert d["ell"].min() >= 0
    assert d["ell"].max() <= 10.0
    assert d["kappa"].abs().max() <= 5.0
    # Average total gain power should be near 1 (with linear dB decay + normalization)
    total_power = (d["h"].abs() ** 2).sum(dim=-1).mean().item()
    assert abs(total_power - 1.0) < 0.05, f"average total path power = {total_power}"


def test_tdl_profile_shape():
    tdl = TDLProfile(profile="TDL-C", delay_spread_ns=300, delta_f_hz=15e3,
                     doppler_hz=500, P_use=5, device=DEVICE)
    d = tdl.sample(batch=4, N=128)
    assert d["ell"].shape == (4, 5)
    assert d["kappa"].shape == (4, 5)
    assert d["h"].shape == (4, 5)


def test_channel_apply_preserves_shape():
    N, N_cp = 64, 8
    ch = UniformFractionalChannel(P=3, ell_max=6.0, kappa_max=2.0, device=DEVICE)
    dd = DoublyDispersiveChannel(N=N, N_cp=N_cp, device=DEVICE)
    d = ch.sample(batch=2)
    s = torch.randn(2, N + N_cp, dtype=torch.complex64, device=DEVICE)
    r = dd.apply(s, d["ell"], d["kappa"], d["h"])
    assert r.shape == s.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
