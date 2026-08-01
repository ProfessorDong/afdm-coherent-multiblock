"""Verify v2 architectural fixes produce a well-behaved untrained receiver.

Fixes tested:
  1. Zero-init Set-Transformer output_proj (delta = 0 at t=0).
  2. Close-init gate (b << 0, so g ~ 0 at t=0).
  3. Tiny-init LM step size (gamma ~ 0 at t=0).

Under these inits, the untrained receiver should reduce to classical CG.

Config: N=128, N_p=32, P=3 (workable config where classical CG has real headroom
above genie).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    genie_mmse_sweep,
)
from afdm.classical import ClassicalCGDetector


def apply_v2_init(rx, gate_b=-8.0, gamma_raw=-5.0):
    """Apply the v2 architectural inits: zero delta, closed gate, tiny LM step."""
    for layer in rx.layers:
        # Zero Set-Transformer output projection
        layer.set_transformer.output_proj.weight.data.zero_()
        layer.set_transformer.output_proj.bias.data.zero_()
        # Close gate
        layer.gate.b.data.fill_(gate_b)
        # Tiny LM step size (support essentially frozen initially)
        layer.gamma_raw.data.fill_(gamma_raw)
    return rx


def main():
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=3, seed=0,
    )
    snrs = [5.0, 15.0, 25.0]
    n_batches, batch_size = 4, 32

    print("=" * 80)
    print("V2 INIT VERIFICATION")
    print(f"Config: N={cfg.N}, N_p={cfg.N_p}, P={cfg.P}, P_max={cfg.P_max} (workable)")
    print("=" * 80)

    # 1. Genie MMSE bound
    genie = genie_mmse_sweep(cfg, snrs, n_batches, batch_size, seed=42)

    # 2. Classical CG
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    classical_res = evaluate_classical_sweep(classical, cfg, snrs, n_batches, batch_size, seed=42)

    # 3. Untrained UGVEMReceiver (default init) — expected to be BAD
    torch.manual_seed(0)
    rx_default = cfg.receiver()
    default_res = evaluate_receiver_sweep(rx_default, cfg, snrs, n_batches, batch_size, seed=42)

    # 4. Untrained + v2 init — should MATCH classical CG
    torch.manual_seed(0)
    rx_v2 = apply_v2_init(cfg.receiver())
    v2_res = evaluate_receiver_sweep(rx_v2, cfg, snrs, n_batches, batch_size, seed=42)

    print(f"\n{'Configuration':<40s}  " + "  ".join(f"{s:>9.1f}dB" for s in snrs))
    print("-" * 82)
    for name, res, keyname in [
        ("1. Genie MMSE (bound)",             genie,        "genie"),
        ("2. Classical CG",                    classical_res, "class"),
        ("3. Untrained receiver (default)",   default_res,  "def"),
        ("4. Untrained receiver (v2 init)",   v2_res,       "v2"),
    ]:
        sers = "  ".join(f"{res[s]['ser']:>10.3e}" for s in snrs)
        print(f"{name:<40s}  {sers}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    for snr in snrs:
        cls_ser = classical_res[snr]["ser"]
        v2_ser = v2_res[snr]["ser"]
        def_ser = default_res[snr]["ser"]
        matching = "MATCH" if abs(v2_ser - cls_ser) / cls_ser < 0.25 else "DIVERGE"
        print(f"  @ {snr:>4.1f} dB: default {def_ser:.2e} -> v2 init {v2_ser:.2e}  vs classical {cls_ser:.2e}  [{matching}]")


if __name__ == "__main__":
    main()
