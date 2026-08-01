"""Tests for Set Transformer and Uncertainty Gate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm.set_transformer import SetTransformer, UncertaintyGate, SetTransformerBlock


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_set_transformer_output_shape():
    B, P, d_in = 4, 6, 5
    m = SetTransformer(input_dim=d_in).to(DEVICE)
    x = torch.randn(B, P, d_in, device=DEVICE)
    out = m(x)
    assert out.shape == (B, P), f"got {out.shape}"
    assert out.dtype in (torch.complex64, torch.complex128)


def test_set_transformer_permutation_equivariance():
    """Permuting input tokens should permute output tokens correspondingly."""
    torch.manual_seed(0)
    B, P, d_in = 3, 5, 5
    m = SetTransformer(input_dim=d_in, d_model=32, n_heads=2, n_blocks=2).to(DEVICE)
    m.eval()
    x = torch.randn(B, P, d_in, device=DEVICE)
    with torch.no_grad():
        out_original = m(x)
        # Permute path axis (dim=1)
        perm = torch.tensor([2, 0, 4, 1, 3], device=DEVICE)
        out_permuted = m(x[:, perm, :])
    # Check that out_permuted equals out_original[:, perm]
    max_err = (out_permuted - out_original[:, perm]).abs().max().item()
    assert max_err < 1e-4, f"permutation equivariance violated: max error {max_err}"


def test_set_transformer_masking_zeros_padding():
    """Padded positions should have zero output."""
    torch.manual_seed(0)
    m = SetTransformer(input_dim=5, d_model=32).to(DEVICE)
    m.eval()
    x = torch.randn(2, 6, 5, device=DEVICE)
    mask = torch.tensor([[True]*4 + [False]*2, [True]*5 + [False]*1], device=DEVICE)
    with torch.no_grad():
        out = m(x, mask=mask)
    # Padded positions should be exactly zero.
    assert (out[~mask].abs() < 1e-6).all(), \
        f"padded positions not zeroed: {out[~mask]}"


def test_set_transformer_norm_clipping():
    """Output norm per node should be <= max_delta_norm."""
    torch.manual_seed(0)
    max_norm = 2.0
    m = SetTransformer(input_dim=5, d_model=32, max_delta_norm=max_norm).to(DEVICE)
    # Bloat weights so unclipped output would be huge.
    with torch.no_grad():
        for p in m.output_proj.parameters():
            p.mul_(100.0)
    x = torch.randn(4, 6, 5, device=DEVICE)
    with torch.no_grad():
        out = m(x)
    per_node_norm = out.abs()
    assert (per_node_norm <= max_norm + 1e-4).all(), \
        f"norms exceed clip: max={per_node_norm.max()}"


def test_uncertainty_gate_shapes():
    g = UncertaintyGate(u_ref=1e-3).to(DEVICE)
    v = torch.rand(4, 6, device=DEVICE)  # positive
    out = g(v)
    assert out.shape == (4,)
    assert (out > 0).all() and (out < 1).all()


def test_uncertainty_gate_monotone_in_variance():
    """gate(v_large) > gate(v_small) for componentwise-increasing v."""
    g = UncertaintyGate(u_ref=1e-3).to(DEVICE)
    v_small = torch.full((1, 6), 1e-4, device=DEVICE)
    v_large = torch.full((1, 6), 1.0, device=DEVICE)
    assert g(v_large).item() > g(v_small).item()


def test_uncertainty_gate_vanishing_at_high_snr():
    """As v -> 0, gate should -> 0. This is the key architectural property (Theorem 2)."""
    g = UncertaintyGate(u_ref=1e-2, init_a=1.0, init_b=0.0).to(DEVICE)
    for v_val in [1e-3, 1e-6, 1e-10, 1e-14]:
        v = torch.full((1, 6), v_val, device=DEVICE)
        val = g(v).item()
        print(f"  v={v_val:.1e}: gate={val:.6f}")
    # At v=1e-14 the gate should be very small
    v_min = torch.full((1, 6), 1e-14, device=DEVICE)
    assert g(v_min).item() < 0.1, f"gate did not close: {g(v_min).item()}"


def test_uncertainty_gate_positive_slope_after_gradient_step():
    """After random gradient updates, the effective slope a = softplus(tilde_a) should remain positive."""
    torch.manual_seed(0)
    g = UncertaintyGate(u_ref=1e-2).to(DEVICE)
    opt = torch.optim.Adam(g.parameters(), lr=0.1)
    # Random loss to trigger updates
    for _ in range(10):
        v = torch.rand(4, 6, device=DEVICE)
        loss = (g(v) - 0.5).pow(2).mean()  # target 0.5
        opt.zero_grad(); loss.backward(); opt.step()
    a = torch.nn.functional.softplus(g.tilde_a).item()
    assert a > 0, f"a became non-positive: {a}"


def test_gate_backprop_flows_to_v():
    """Gradient w.r.t. v should be non-zero (v enters through log)."""
    g = UncertaintyGate(u_ref=1e-2).to(DEVICE)
    v = torch.rand(4, 6, device=DEVICE, requires_grad=True)
    out = g(v).sum()
    (grad_v,) = torch.autograd.grad(out, v)
    assert grad_v.abs().max().item() > 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
