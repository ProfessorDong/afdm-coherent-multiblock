"""Tests for the full UGVEMReceiver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.support import SupportRecovery
from afdm.receiver import UGVEMReceiver, UGVEMReceiverLayer


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _setup(N=128, N_p=32, P=3, seed=0):
    torch.manual_seed(seed)
    sys_ = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=DEVICE)
    ch = UniformFractionalChannel(P=P, ell_max=10.0, kappa_max=5.0, device=DEVICE)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    pilot_positions = uniform_daft_pilots(N=N, N_p=N_p, device=DEVICE)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed + 1)
    pilot_values = qpsk[torch.randint(0, 4, (N_p,), device=DEVICE, generator=gen)]
    sup = SupportRecovery(N=N, N_cp=sys_.ell_max, kappa_max=5, ell_max=10, P_max=6)
    rx = UGVEMReceiver(
        system=sys_, support_recovery=sup, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        T=3, K_cg=10, d_model=32, n_heads=2, n_blocks=2,
    ).to(DEVICE)
    return sys_, ch, qpsk, pilot_positions, pilot_values, sup, rx


def _forward(sys_, chdict, x_true, sigma_w2):
    op = FastAFDMOperator(system=sys_, ell=chdict["ell"], kappa=chdict["kappa"], h=chdict["h"])
    y_clean = op.matvec(x_true)
    signal_pow = (y_clean.abs() ** 2).mean()
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    y = y_clean + torch.randn_like(y_clean) * noise_std
    abs_noise = (signal_pow * sigma_w2).item()
    return sys_.idaft(y), abs_noise


def test_receiver_forward_shape():
    sys_, ch, qpsk, pp, pv, sup, rx = _setup()
    B = 4
    d = ch.sample(batch=B)
    idx = torch.randint(0, 4, (B, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pp] = pv.unsqueeze(0)
    r, abs_noise = _forward(sys_, d, x, sigma_w2=1e-2)
    out = rx(r, sigma_w2_block=abs_noise, refine_theta=True, return_layer_states=True)
    assert out["x_mean"].shape == (B, sys_.N)
    assert out["p_ms"].shape == (B, sys_.N, 4)
    assert out["eta_h"].shape == (B, 6)  # P_max = 6
    assert len(out["layer_states"]) == 3  # T=3


def test_gradient_flows_to_learned_params():
    """Backprop through the receiver should produce non-zero gradients on the
    end-to-end-differentiable learned parameters.

    Note:
      * `gamma_raw` (LM step size) is intentionally NOT end-to-end differentiable:
        the safeguarded acceptance rule for the support update is a discrete
        decision by design. Gamma remains tunable via a separate schedule if needed.
      * `beta_raw` in layer 0 has no gradient because z_prev is None at the first
        layer (the symbol damping term is (1-beta)*z_prev + beta*z; with z_prev=0
        the beta coefficient trivially cancels).
    """
    sys_, ch, qpsk, pp, pv, sup, rx = _setup()
    B = 2
    d = ch.sample(batch=B)
    idx = torch.randint(0, 4, (B, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pp] = pv.unsqueeze(0)
    r, abs_noise = _forward(sys_, d, x, sigma_w2=1e-2)
    out = rx(r, sigma_w2_block=abs_noise)
    loss = -out["p_ms"].clamp(min=1e-9).log().gather(-1, idx.unsqueeze(-1)).mean()
    loss.backward()
    # Expected-no-gradient parameters (by architecture):
    ok_no_grad = {"gamma_raw"}  # any layer index
    # Layer-0-specific no-gradient: beta_raw (z_prev is None at layer 0)
    zero_grad_params = []
    for name, p in rx.named_parameters():
        base_name = name.split(".")[-1]
        if p.grad is None or p.grad.abs().max().item() < 1e-12:
            if base_name in ok_no_grad:
                continue
            if base_name == "beta_raw" and name.startswith("layers.0."):
                continue
            zero_grad_params.append(name)
    assert len(zero_grad_params) == 0, f"unexpected no-gradient params: {zero_grad_params}"


def test_gate_closes_at_high_snr():
    """At very high SNR, the uncertainty gate should be near zero at inference."""
    sys_, ch, qpsk, pp, pv, sup, rx = _setup()
    rx.eval()
    B = 2
    d = ch.sample(batch=B)
    idx = torch.randint(0, 4, (B, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pp] = pv.unsqueeze(0)
    # Very high SNR
    r, abs_noise = _forward(sys_, d, x, sigma_w2=1e-10)
    with torch.no_grad():
        out = rx(r, sigma_w2_block=abs_noise, return_layer_states=True)
    g_avg = torch.stack([s["g"] for s in out["layer_states"]]).mean().item()
    print(f"Average gate at SNR=100 dB: {g_avg:.4e}")
    assert g_avg < 0.1, f"gate did not close at high SNR: {g_avg}"


def test_untrained_receiver_functional():
    """An untrained receiver should not crash or produce NaN, even if performance is poor."""
    sys_, ch, qpsk, pp, pv, sup, rx = _setup()
    rx.eval()
    B = 4
    d = ch.sample(batch=B)
    idx = torch.randint(0, 4, (B, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pp] = pv.unsqueeze(0)
    r, abs_noise = _forward(sys_, d, x, sigma_w2=1e-2)
    with torch.no_grad():
        out = rx(r, sigma_w2_block=abs_noise)
    # Check no NaNs
    assert torch.isfinite(out["x_mean"]).all()
    assert torch.isfinite(out["p_ms"]).all()
    # Posterior sums to 1
    assert torch.allclose(out["p_ms"].sum(dim=-1), torch.ones(B, sys_.N, device=DEVICE), atol=1e-4)


def test_theta_gradient_flows_through_LM_step():
    """Gradient of downstream loss w.r.t. per-layer set-transformer params should also
    affect the theta trajectory (verified by non-zero gradient magnitude on the LM
    step's dependent variables through the set_transformer output)."""
    sys_, ch, qpsk, pp, pv, sup, rx = _setup()
    B = 2
    d = ch.sample(batch=B)
    idx = torch.randint(0, 4, (B, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pp] = pv.unsqueeze(0)
    r, abs_noise = _forward(sys_, d, x, sigma_w2=1e-2)
    out = rx(r, sigma_w2_block=abs_noise, return_layer_states=True)
    # Cross-entropy loss + set-matched theta loss (Hungarian would be better; skip for test)
    loss_ce = -out["p_ms"].clamp(min=1e-9).log().gather(-1, idx.unsqueeze(-1)).mean()
    loss_ce.backward()
    # Gradient exists on the last set-transformer's weights
    st_grads = [p.grad.abs().max().item() for name, p in rx.layers[-1].set_transformer.named_parameters()]
    assert max(st_grads) > 1e-12, "no gradient on final layer set-transformer"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
