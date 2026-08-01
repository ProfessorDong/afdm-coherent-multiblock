# AFDM_TCOM code

Reference implementation for *Uncertainty-Gated Variational-EM Unfolding for Joint
Off-Grid Channel Estimation and Detection in AFDM* (Dong, IEEE TCOM submission).

## Structure

```
afdm/
  system.py       AFDM Tx/Rx primitives: DAFT, IDAFT, CPP
  channels.py     Doubly-dispersive channels: uniform, 3GPP TDL profiles
  operators.py    Fast quasi-banded DAFT-domain operator
  pilots.py       Deterministic pilot patterns
tests/            Unit tests (pytest)
scripts/          End-to-end smoke tests and figure regeneration
```

## Conventions

- Complex tensors are `torch.complex64` on GPU (`cuda:0`, RTX 4090).
- Leading dimension is always batch (`B`), followed by symbol/subcarrier index (`N`).
- Delay in units of samples (fractional). Doppler in normalized-index units $\kappa = \nu / \Delta f$ (fractional).
- Path count `P` is per-block and may differ between generation and detection.

## Development phases

Per `IMPLEMENTATION_PLAN.md`:
- **P1 (current)**: Foundation — AFDM system, channels, fast operator, unit tests.
- **P2**: Baselines — classical CG, PBiGaBP, JPNCE-SBL, off-grid support recovery.
- **P3**: Proposed receiver — V-EM iterations, gated Set-Transformer, LM support.
- **P4**: Experiments and figure regeneration.
- **P5**: 5G-NR LDPC coded BLER.
- **P6**: Uncertainty calibration.
