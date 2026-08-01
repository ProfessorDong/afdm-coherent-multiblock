# Implementation plan — AFDM_TCOM reference code

Total scope: ~8 weeks focused effort to produce publication-quality evidence
for the TCOM submission. All heavy computation runs on `cuda:0` (RTX 4090).

## Status

- **P1 Foundation — COMPLETE.** AFDM system (DAFT/IDAFT/CPP), doubly-dispersive
  channel with fractional delay-Doppler, fast quasi-banded operator with
  truncated-FIR fractional shifts, deterministic pilot patterns. 17 unit tests
  pass; end-to-end genie-CSI smoke test reaches BER 0 by SNR = 25 dB on QPSK.

- **P2 Baselines — COMPLETE.** Off-grid support recovery (CFAR + Newton on
  integer-delay ambiguity + oversampled Doppler); classical semi-blind CG detector;
  PBiGaBP receiver (message-passing Bayesian h + soft symbols + optional
  safeguarded theta refinement); JPNCE-SBL receiver (ARD prior + EM + CFAR-init
  candidates + dynamic-grid updates + magnitude-based pruning). 29 unit tests
  pass; smoke test shows all baselines floor at 20-30% SER while genie CG-MMSE
  reaches 0% by 25 dB — exactly the floor phenomenon the proposed learned
  receiver targets.

- **P3 Proposed receiver — pending.** V-EM iterations; damped exact ridge for
  h; safeguarded Levenberg-Marquardt for theta; Set-Transformer path attention;
  uncertainty gate; safeguarded acceptance; training loop with Hungarian set
  loss.

- **P4 Experiments + figures — pending.** All 10 paper figures + confidence
  intervals over 3 training seeds.

- **P5 Coded BLER — pending.** 5G-NR LDPC integration (BG1, rate 1/2 & 3/4);
  coded-BLER-vs-SNR figure.

- **P6 Uncertainty calibration — pending.** Reliability diagram (predicted std
  vs empirical squared error); ECE metric; gate-value-vs-SNR distribution.

## Directory layout

```
code/
├── README.md
├── requirements.txt
├── IMPLEMENTATION_PLAN.md         (this file)
├── afdm/                          Core library
│   ├── system.py                    DAFT/IDAFT/CPP  [DONE]
│   ├── channels.py                  Uniform/TDL, complex Dirichlet kernel  [DONE]
│   ├── operators.py                 FastAFDMOperator + slow reference  [DONE]
│   ├── pilots.py                    Deterministic pilot patterns  [DONE]
│   ├── support.py                   CFAR + Newton off-grid support  [DONE]
│   ├── classical.py                 Algorithm 1 (semi-blind CG)  [DONE]
│   ├── pbigabp.py                   Ranasinghe et al. 2025 baseline  [DONE]
│   ├── jpnce_sbl.py                 Xu et al. 2026 baseline  [DONE]
│   ├── set_transformer.py           Path-attention + uncertainty gate  [DONE]
│   ├── vem.py                       V-EM primitives (h/theta/x updates)  [DONE]
│   ├── receiver.py                  UGVEMReceiver, T-layer forward + backward  [DONE]
│   ├── loss.py                      Hungarian-matched set loss + anchor + CE  [DONE]
│   ├── training.py                  Training loop, seed control, val metrics  [DONE]
│   ├── ldpc.py                      5G-NR LDPC encoder/decoder wrapper  [P5]
│   └── calibration.py               Reliability diagrams, ECE  [P6]
├── tests/                         pytest suite
│   ├── test_daft.py                 [DONE 8/8 pass]
│   ├── test_channel.py              [DONE 5/5 pass]
│   ├── test_fast_operator.py        [DONE 4/4 pass]
│   ├── test_support.py              [DONE 4/4 pass]
│   ├── test_classical.py            [DONE 4/4 pass]
│   ├── test_pbigabp.py              [DONE 2/2 pass]
│   ├── test_jpnce_sbl.py            [DONE 2/2 pass]
│   ├── test_set_transformer.py      [DONE 9/9 pass]
│   ├── test_vem.py                  [DONE 8/8 pass]
│   ├── test_receiver.py             [DONE 5/5 pass]
│   ├── test_loss.py                 [DONE 5/5 pass]
│   ├── test_vem.py                  [P3]
│   ├── test_gate.py                 [P3]
│   ├── test_loss.py                 [P3]
│   ├── test_ldpc.py                 [P5]
│   └── test_calibration.py          [P6]
└── scripts/                       End-to-end experiments and figure regen
    ├── verify_p1_smoke.py           [DONE]
    ├── verify_p2_smoke.py           [DONE]
    ├── verify_p3_smoke.py           [DONE]
    ├── run_ber_vs_snr.py            [P4]
    ├── run_nmse_vs_snr.py           [P4]
    ├── run_delay_doppler_rmse.py    [P4]
    ├── run_ablation.py              [P4]
    ├── run_convergence.py           [P4]
    ├── run_ber_vs_doppler.py        [P4]
    ├── run_delay_sensitivity.py     [P4]
    ├── run_phase_noise.py           [P4]
    ├── run_pilot_overhead.py        [P4]
    ├── run_complexity.py            [P4]
    ├── run_coded_bler.py            [P5]
    └── run_calibration.py           [P6]
```

## Conventions established in P1

- Complex tensors are `torch.complex64` on `cuda:0`.
- Leading dim is batch; last dim is subcarrier/symbol index.
- Delay in samples (fractional); Doppler in normalized units kappa = nu/Delta_f.
- Fractional-shift kernel: complex Dirichlet with phase factor
  exp(j pi (N-1) (m - delta) / N), consistent with FFT-based fractional shifts
  using k = 0, 1, ..., N-1 (non-negative convention).
- Chirp params: c_1 = (2 kappa_max + 1)/(2N), c_2 = 1/(2N); N must be even so
  the CPP reduces to an ordinary CP.
- DAFT is unitary, and the DAFT-domain channel operator H^D is exactly
  represented by `FastAFDMOperator`, which is the model used by both the
  physical channel simulation and the receiver (they are the same object).

## Bug discovered and documented for the paper

`DoublyDispersiveChannel.apply` (the time-domain channel simulator) uses
FFT-based fractional shifts within the length-(N + N_cp) buffer, whereas the
correct physical model has delay circular within length-N (post-CP-strip).
For all training and evaluation, `FastAFDMOperator` is the canonical forward
model. `DoublyDispersiveChannel` is retained only for pedagogical /
verification purposes and its `apply` method will not be used in the main
pipeline.

## P2 lessons learned (documented for P3)

1. **Dirichlet kernel convention matters.** The complex phase-included periodic-sinc
   is required for consistency between FFT-based fractional shifts and Toeplitz
   convolution; the magnitude-only Dirichlet common in DSP textbooks is inconsistent.
2. **Signed-k vs non-negative-k conventions differ for fractional shifts.** We use
   k = 0..N-1 (non-negative) everywhere.
3. **Ambiguity-function fractional-delay interpolation via zero-pad IFFT uses
   signed-k, so it is inconsistent with the non-negative-k fast operator.** Use
   integer-delay ambiguity + Newton refinement, not oversampled-delay
   interpolation.
4. **Gradient-based support refinement (as in PBiGaBP) is unstable without a
   safeguard.** Even with per-parameter step clipping and per-sample-normalized
   loss, false-positive candidates drift arbitrarily. This directly motivates the
   proposed receiver's safeguarded acceptance rule.
5. **SBL requires well-separated initial candidates to escape rank-deficiency.**
   CFAR + Newton initialization gives far better SBL performance than a dense
   overcomplete grid; SBL on a dense grid can fail catastrophically due to
   candidate correlations.
6. **Baselines all floor at moderate SER** across the SNR range because of
   support-recovery imperfections and the LS/decision-directed h-step's bias.
   This is exactly the floor the proposed uncertainty-gated V-EM receiver
   targets (Theorems 1 and 2 in the paper).

## P3 lessons learned (documented for P4)

1. **Learned scalars must stay tensors through the entire graph.** `.item()` or
   `float()` casts silently detach and prevent gradient flow — a whole class of
   parameters can become "learned" in name only.
2. **The safeguarded LM step is intentionally non-differentiable end-to-end.**
   The accept/reject decision is discrete; gamma_lm cannot receive gradients
   through it. This is the correct architectural trade-off for the paper's
   monotonicity theorem (Theorem 3).
3. **beta_raw in layer 0 receives no gradient because z_prev is None there.**
   Not a bug — the damping formula (1-beta)*z_prev + beta*z has beta*0 + beta*z
   = z when z_prev=0.
4. **Hungarian matching via scipy.optimize.linear_sum_assignment is fast for
   small P (< 20).** Batched via a Python loop; the ~1ms overhead is negligible
   vs the CG-MMSE cost per batch.
5. **Full training convergence takes hundreds of epochs.** The 30-epoch smoke
   test shows loss decreases and SER modestly improves, but the headline SER
   numbers in the paper require the full 500-epoch schedule (P4).
6. **Cardinality-normalized gate (mean rather than sum of variances) is
   essential** for the gate to be invariant across P values. Confirmed by the
   test_gate_closes_at_high_snr passing across variable batch cardinalities.

- **P4 Experiments + figure regeneration — COMPLETE (moderate-preset).**
  All 10 figure-generation scripts (`scripts/run_*.py`), all 4 ablation
  training runs (proposed / gate / attention / scalars), unified experiment
  orchestrator (`afdm/experiments.py`), reproducibility guide (`REPRODUCIBILITY.md`).
  Figures at `code/figures/*.pdf` reflect the `--quick` training preset (20
  epochs); publication numbers require `--publication` (500 epochs, ~24h on
  RTX 4090). 56/56 unit tests still pass after all P4 changes.

## Next milestone (deprecated — original planning below)
- Full-scale training campaign (500 epochs, N=128, T=8) with 3 random seeds.
- `scripts/run_ber_vs_snr.py`: figure 1 (BER vs SNR).
- `scripts/run_nmse_vs_snr.py`: figure 2 (NMSE, showing no high-SNR floor).
- `scripts/run_delay_doppler_rmse.py`: figure 3 (delay/Doppler RMSE).
- `scripts/run_ablation.py`: figure 4 (each learned component independently retrained).
- `scripts/run_convergence.py`: figure 5 (per-layer BER).
- `scripts/run_ber_vs_doppler.py`: figure 6 (Doppler robustness).
- `scripts/run_delay_sensitivity.py`: figure 7.
- `scripts/run_phase_noise.py`: figure 8.
- `scripts/run_pilot_overhead.py`: figure 9.
- `scripts/run_complexity.py`: figure 10 (FLOPs).
- All figures generated with confidence intervals over 3 seeds.
- Update paper text and tables with real numbers.
