"""Rigorous diagnosis of the training plateau.

Question: at the *publication* config (N=128, T=8, P=5, P_max=5) does the
UGVEMReceiver reduce to classical CG when its learned corrections are disabled?
If not, we have an architectural bug that no amount of training will fix.

Runs 6 checkpoints back-to-back:
  1. Classical CG detector (reference lower bound).
  2. Genie CG-MMSE (upper bound).
  3. Untrained UGVEMReceiver (random init).
  4. Untrained UGVEMReceiver with zero-delta (SetTransformer output_proj zeroed).
  5. Untrained with zero-delta AND closed-gate.
  6. Trained UGVEMReceiver from publication run (best.pt).

If (3) is much worse than (1), the untrained receiver is broken.
If (4) matches (1), the SetTransformer is the culprit.
If (5) matches (1), gate + delta together matter.
If (6) is only slightly better than (5), training isn't extracting signal from
the learned components.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import (
    ExperimentConfig, build_ablation, evaluate_receiver_sweep,
    evaluate_classical_sweep, genie_mmse_sweep, load_receiver,
)
from afdm.classical import ClassicalCGDetector


def zero_delta_init(rx):
    """Zero the Set-Transformer output projection so delta=0 at init."""
    for layer in rx.layers:
        layer.set_transformer.output_proj.weight.data.zero_()
        layer.set_transformer.output_proj.bias.data.zero_()
    return rx


def close_gate_init(rx, b_val=-8.0):
    """Set gate.b to a large negative value so g ≈ 0 initially."""
    for layer in rx.layers:
        layer.gate.b.data.fill_(b_val)
    return rx


def freeze_lm_step(rx, gamma_raw=-8.0):
    """Set gamma_raw to a large negative so LM step size ≈ 0 (support frozen)."""
    for layer in rx.layers:
        layer.gamma_raw.data.fill_(gamma_raw)
    return rx


@torch.no_grad()
def evaluate(rx, cfg, snrs, n_batches=4, batch_size=32):
    return evaluate_receiver_sweep(rx, cfg, snrs, n_batches, batch_size, seed=42)


def main():
    device = "cuda:0"
    torch.manual_seed(0)
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=5, seed=0,
    )
    snrs = [5.0, 15.0, 25.0]
    n_batches = 4; batch_size = 32

    print("=" * 80)
    print("PLATEAU DIAGNOSIS")
    print(f"Config: N={cfg.N}, T={cfg.T}, P={cfg.P}, P_max={cfg.P_max}, N_p={cfg.N_p}")
    print(f"SNRs: {snrs} dB; {n_batches} batches × {batch_size} samples each")
    print("=" * 80)

    results = {}

    # 1. Classical CG
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    t0 = time.time()
    results["1. Classical CG"] = evaluate_classical_sweep(classical, cfg, snrs, n_batches, batch_size, seed=42)
    print(f"[{time.time()-t0:.1f}s] Classical done")

    # 2. Genie CG-MMSE
    t0 = time.time()
    results["2. Genie CG-MMSE"] = genie_mmse_sweep(cfg, snrs, n_batches, batch_size, seed=42)
    print(f"[{time.time()-t0:.1f}s] Genie done")

    # 3. Untrained UGVEMReceiver (random init)
    torch.manual_seed(0)
    rx3 = cfg.receiver()
    t0 = time.time()
    results["3. Untrained (random)"] = evaluate(rx3, cfg, snrs, n_batches, batch_size)
    print(f"[{time.time()-t0:.1f}s] Untrained random done")

    # 4. Untrained + zero-delta
    torch.manual_seed(0)
    rx4 = zero_delta_init(cfg.receiver())
    t0 = time.time()
    results["4. Untrained + zero-delta"] = evaluate(rx4, cfg, snrs, n_batches, batch_size)
    print(f"[{time.time()-t0:.1f}s] Zero-delta done")

    # 5. Untrained + zero-delta + closed-gate + frozen LM
    torch.manual_seed(0)
    rx5 = freeze_lm_step(close_gate_init(zero_delta_init(cfg.receiver())))
    t0 = time.time()
    results["5. Untrained + closed gate + no LM"] = evaluate(rx5, cfg, snrs, n_batches, batch_size)
    print(f"[{time.time()-t0:.1f}s] Full-clean done")

    # 6. Trained publication receiver (best.pt from proposed_seed0)
    ckpt = Path("runs/pub_v1/proposed_seed0/best.pt")
    if ckpt.exists():
        state = torch.load(ckpt, weights_only=False, map_location=device)
        rx6 = cfg.receiver()
        rx6.load_state_dict(state["state_dict"])
        t0 = time.time()
        results["6. Trained (pub_v1 best)"] = evaluate(rx6, cfg, snrs, n_batches, batch_size)
        print(f"[{time.time()-t0:.1f}s] Trained done")

    print("\n" + "=" * 80)
    print("RESULTS: SER at each SNR")
    print("=" * 80)
    header = f"{'Configuration':<38s}  " + "  ".join(f"{s:>9.1f}dB" for s in snrs)
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        sers = "  ".join(f"{res[s]['ser']:>10.3e}" for s in snrs)
        print(f"{name:<38s}  {sers}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    classical_15 = results["1. Classical CG"][15.0]["ser"]
    genie_15 = results["2. Genie CG-MMSE"][15.0]["ser"]
    untrained_15 = results["3. Untrained (random)"][15.0]["ser"]
    zerodelta_15 = results["4. Untrained + zero-delta"][15.0]["ser"]
    clean_15 = results["5. Untrained + closed gate + no LM"][15.0]["ser"]
    print(f"At 15 dB:")
    print(f"  Genie MMSE (bound):                    {genie_15:.3e}")
    print(f"  Classical CG (baseline to beat):       {classical_15:.3e}")
    print(f"  Untrained random UGVEM:                {untrained_15:.3e}  {'(BROKEN' if untrained_15 > 2*classical_15 else '(OK'})")
    print(f"  Untrained + zero-delta:                {zerodelta_15:.3e}  {'(BROKEN' if zerodelta_15 > 2*classical_15 else '(OK'})")
    print(f"  Untrained + clean init:                {clean_15:.3e}  {'(BROKEN' if clean_15 > 2*classical_15 else '(OK'})")
    if "6. Trained (pub_v1 best)" in results:
        trained_15 = results["6. Trained (pub_v1 best)"][15.0]["ser"]
        print(f"  Trained (pub_v1 best):                 {trained_15:.3e}")
        gain = classical_15 / max(trained_15, 1e-12)
        print(f"  Trained vs Classical improvement:      {gain:.2f}x")


if __name__ == "__main__":
    main()
