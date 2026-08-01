"""Unit tests for support recovery: ambiguity, CFAR, Newton refinement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.support import ambiguity_function, cfar_peaks, newton_refine, SupportRecovery


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_pilot_only_signal(N: int, N_p: int, device: str) -> tuple[torch.Tensor, AFDMSystem]:
    """Build a time-domain pilot-only transmit signal."""
    sys_ = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=device)
    pilot_pos = uniform_daft_pilots(N=N, N_p=N_p, device=device)
    qpsk = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], device=device, dtype=torch.complex64) / (2 ** 0.5)
    # Pilot values (deterministic seed).
    gen = torch.Generator(device=device); gen.manual_seed(0)
    idx = torch.randint(0, 4, (N_p,), device=device, generator=gen)
    x = torch.zeros(N, dtype=torch.complex64, device=device)
    x[pilot_pos] = qpsk[idx]
    s = sys_.idaft(x.unsqueeze(0))[0]  # (N,) time-domain
    return s, sys_


def test_ambiguity_peaks_at_true_support_single_path():
    """With a single-path noiseless channel, ambiguity should peak at the true (ell, kappa)."""
    N, N_cp = 128, 10
    torch.manual_seed(0)
    s, sys_ = _build_pilot_only_signal(N=N, N_p=32, device=DEVICE)
    # Single path: ell=3.4, kappa=1.7
    ell_true, kappa_true = 3.4, 1.7
    ell = torch.tensor([[ell_true]], device=DEVICE)
    kappa = torch.tensor([[kappa_true]], device=DEVICE)
    h = torch.tensor([[1.0 + 0j]], device=DEVICE, dtype=torch.complex64)
    op = FastAFDMOperator(system=sys_, ell=ell, kappa=kappa, h=h)
    y = op.matvec(sys_.daft(s.unsqueeze(0)))  # (1, N) DAFT-domain
    r = sys_.idaft(y)  # back to time domain

    A, ell_grid, kappa_grid = ambiguity_function(
        r=r, s_pilot=s, N=N, N_cp=N_cp, kappa_max=5, ell_max=10,
        oversample_doppler=2,
    )
    peak_flat = A[0].argmax()
    k_idx = peak_flat // A.shape[-1]
    l_idx = peak_flat % A.shape[-1]
    ell_peak = ell_grid[l_idx].item()
    kap_peak = kappa_grid[k_idx].item()
    print(f"Peak at ell={ell_peak:.2f} (true {ell_true}), kappa={kap_peak:.2f} (true {kappa_true})")
    # Integer-delay peak should be within 1 of the true delay; Doppler grid step is 0.45.
    assert abs(ell_peak - ell_true) <= 1.0, f"delay peak off: {ell_peak} vs {ell_true}"
    assert abs(kap_peak - kappa_true) <= 1.0, f"Doppler peak off: {kap_peak} vs {kappa_true}"


def test_newton_refines_toward_true_support():
    """After Newton refinement, the fractional peak should be much closer to the truth."""
    N, N_cp = 128, 10
    torch.manual_seed(0)
    s, sys_ = _build_pilot_only_signal(N=N, N_p=32, device=DEVICE)
    ell_true, kappa_true = 3.4, 1.7
    ell = torch.tensor([[ell_true]], device=DEVICE)
    kappa = torch.tensor([[kappa_true]], device=DEVICE)
    h = torch.tensor([[1.0 + 0j]], device=DEVICE, dtype=torch.complex64)
    op = FastAFDMOperator(system=sys_, ell=ell, kappa=kappa, h=h)
    y = op.matvec(sys_.daft(s.unsqueeze(0)))
    r = sys_.idaft(y)

    A, ell_grid, kappa_grid = ambiguity_function(
        r=r, s_pilot=s, N=N, N_cp=N_cp, kappa_max=5, ell_max=10,
        oversample_doppler=2,
    )
    peak_idx, _ = cfar_peaks(A, K=1)
    ell_ref, kappa_ref = newton_refine(A, peak_idx, ell_grid, kappa_grid, max_iter=1)
    print(f"Refined ell={ell_ref[0, 0].item():.3f} (true {ell_true}), kappa={kappa_ref[0, 0].item():.3f} (true {kappa_true})")
    # Newton on integer grid: with step 1 in delay, quadratic-fit precision should
    # bring error well below 0.5. Doppler oversampled at 0.45 step: precision <=0.25.
    assert abs(ell_ref[0, 0].item() - ell_true) <= 0.3, f"delay refine off: {ell_ref[0, 0].item()}"
    assert abs(kappa_ref[0, 0].item() - kappa_true) <= 0.3, f"Doppler refine off: {kappa_ref[0, 0].item()}"


def test_cfar_finds_multiple_paths():
    """With three separated paths, CFAR should find all three."""
    N, N_cp = 128, 10
    torch.manual_seed(0)
    s, sys_ = _build_pilot_only_signal(N=N, N_p=32, device=DEVICE)
    ell = torch.tensor([[1.0, 5.0, 8.0]], device=DEVICE)
    kappa = torch.tensor([[-2.0, 0.5, 3.0]], device=DEVICE)
    h = torch.tensor([[1.0 + 0j, 0.7 + 0.3j, 0.5 - 0.1j]], device=DEVICE, dtype=torch.complex64)
    op = FastAFDMOperator(system=sys_, ell=ell, kappa=kappa, h=h)
    y = op.matvec(sys_.daft(s.unsqueeze(0)))
    r = sys_.idaft(y)

    sup = SupportRecovery(N=N, N_cp=N_cp, kappa_max=5, ell_max=10, P_max=5)
    ell_hat, kappa_hat, p_hat = sup(r, s)
    print(f"P_hat={p_hat.item()}, ell_hat={ell_hat[0, :p_hat.item()].tolist()}, "
          f"kappa_hat={kappa_hat[0, :p_hat.item()].tolist()}")
    # Should find at least 3 peaks (the true ones).
    assert p_hat.item() >= 3, f"expected >=3 peaks, got {p_hat.item()}"
    # Each true path should have a nearby detection (within 0.5 grid step tolerance).
    ell_hat_np = ell_hat[0, :p_hat.item()].tolist()
    kappa_hat_np = kappa_hat[0, :p_hat.item()].tolist()
    for e_true, k_true in zip([1.0, 5.0, 8.0], [-2.0, 0.5, 3.0]):
        # Find closest detected peak
        dists = [((e - e_true) ** 2 + (k - k_true) ** 2) for e, k in zip(ell_hat_np, kappa_hat_np)]
        min_dist = min(dists)
        assert min_dist < 1.0, f"true path ({e_true}, {k_true}) not detected within 1.0"


def test_noiseless_high_snr_support():
    """At infinite SNR with a random uniform-fractional channel, support recovery should be accurate."""
    N, N_cp, P = 128, 10, 5
    torch.manual_seed(1)
    s, sys_ = _build_pilot_only_signal(N=N, N_p=32, device=DEVICE)
    ch = UniformFractionalChannel(P=P, ell_max=10.0, kappa_max=5.0, device=DEVICE)
    d = ch.sample(batch=4)
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y = op.matvec(sys_.daft(s.unsqueeze(0).expand(4, -1)))
    r = sys_.idaft(y)

    sup = SupportRecovery(N=N, N_cp=N_cp, kappa_max=5, ell_max=10, P_max=P + 2)
    ell_hat, kappa_hat, p_hat = sup(r, s.unsqueeze(0).expand(4, -1))
    # On average, should detect at least P/2 of the true paths.
    for b in range(4):
        p_hat_b = p_hat[b].item()
        matched = 0
        for e_true, k_true in zip(d["ell"][b], d["kappa"][b]):
            for i in range(p_hat_b):
                if abs(ell_hat[b, i] - e_true) < 0.5 and abs(kappa_hat[b, i] - k_true) < 0.5:
                    matched += 1
                    break
        assert matched >= P // 2, f"batch {b}: matched {matched}/{P}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
