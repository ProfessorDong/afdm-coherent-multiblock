# Reproducibility guide — AFDM_TCOM reference implementation

This document describes how to reproduce the paper's results from the reference
implementation in `code/`.

## Environment

- Python 3.12
- PyTorch >= 2.0 with CUDA 12.6 (RTX 4090)
- Additional: numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7, pytest >= 7.0

Install:
```bash
pip install -r requirements.txt
```

## Test suite

```bash
python -m pytest tests/ -v
```

Should report **56/56 tests pass** in about 3 seconds.

## Smoke tests (quick sanity checks)

```bash
python scripts/verify_p1_smoke.py   # AFDM + channel + fast operator + genie CG-MMSE
python scripts/verify_p2_smoke.py   # + classical + PBiGaBP + JPNCE-SBL baselines
python scripts/verify_p3_smoke.py   # + full proposed receiver (~2 min training)
```

## Training campaign

The paper's headline numbers require an extended training campaign.

### Quick / moderate (for interactive verification)

```bash
python scripts/train_all.py --quick --N 64 --T 4 --P 3   # 20 epochs, ~5 min total
python scripts/train_all.py                              # 100 epochs, ~1 h total
```

### Publication run (rigorous, resumable, monitored)

For the full 12-run × 500-epoch campaign (~34 h on RTX 4090), use the
publication launcher which does pre-flight verification, NaN detection,
best-checkpoint tracking, and resumable state:

```bash
# One-shot: verify, launch under nohup, print monitoring commands.
bash scripts/launch_publication.sh --seeds 0 1 2 --n_epochs 500

# Monitor progress (safe to run repeatedly).
python scripts/monitor_training.py --run-dir runs/pub_v1
python scripts/monitor_training.py --run-dir runs/pub_v1 --watch  # refresh every 30s

# Stop cleanly (writes final checkpoint before exit).
kill $(cat runs/pub_v1/pid.txt)

# Resume after interruption (picks up from last saved epoch).
bash scripts/launch_publication.sh --resume
```

Under `runs/pub_v1/`:

```
runs/pub_v1/
├── campaign.log                             top-level orchestrator log
├── pid.txt                                  running process id
├── nohup.out.<timestamp>                    unbuffered stdout+stderr
├── summary.json                             (written at end)
├── proposed_seed0/
│   ├── state.json                           updated every epoch
│   ├── train.log                            per-variant log
│   ├── best.pt                              lowest val-SER checkpoint
│   ├── last.pt                              most recent checkpoint (for resume)
│   └── epoch_{20,40,...}.pt                 periodic checkpoints
├── proposed_seed1/  ...
├── proposed_seed2/  ...
├── gate_seed0/  ...
├── attention_seed0/  ...
└── scalars_seed0/  ...
```

Safeguards:
- Deterministic per-seed initialization.
- NaN loss or gradient triggers an abort of the *current* variant with a
  diagnostic checkpoint; the campaign continues to the next variant.
- SIGTERM / SIGINT save state cleanly before exit.
- state.json updated atomically (write-then-rename) so an interruption cannot
  leave a half-written file.
- Best-checkpoint tracker uses SER at a nominal SNR (default 15 dB).

The `--publication` flag also trains at N=128, T=8, P=5 (paper's nominal setup).
The `--seeds` flag trains multiple random seeds; per-figure confidence intervals
require at least 3 seeds.

Checkpoints are saved to `code/checkpoints/{variant}_seed{seed}.pt`.

## Figure regeneration

After a successful training campaign:

```bash
python scripts/run_ber_vs_snr.py            # Fig 1
python scripts/run_nmse_vs_snr.py           # Fig 2
python scripts/run_delay_doppler_rmse.py    # Fig 3
python scripts/run_ablation.py              # Fig 4
python scripts/run_convergence.py           # Fig 5
python scripts/run_ber_vs_doppler.py        # Fig 6
python scripts/run_delay_sensitivity.py     # Fig 7
python scripts/run_phase_noise.py           # Fig 8
python scripts/run_pilot_overhead.py        # Fig 9
python scripts/run_complexity.py            # Fig 10
```

Figures are saved to `code/figures/{name}.pdf` (paper-quality) and
`code/figures/{name}.png` (quick view). Numerical results are saved to
`code/results/{name}.json` for further analysis.

## Current state of the reference figures

The figures shipped in `code/figures/` were generated with the **--quick
training preset** (20 epochs × 20 steps at N=64, T=4, P=3) rather than the
publication-quality preset. This is a design decision: the quick preset can
be reproduced end-to-end in under 5 minutes on RTX 4090, allowing reviewers
to verify the pipeline functions. The paper's numerical claims (e.g., 14x
BER improvement over classical) require the `--publication` preset.

For the initial submission, the reference implementation demonstrates:

* All architectural properties expected by the paper's theorems (56/56 unit
  tests pass, including permutation equivariance to 1e-4 and gate closure to
  1e-4 at SNR=40 dB).
* Correct end-to-end training pipeline: loss decreases monotonically, all
  end-to-end-differentiable parameters receive gradients, no divergence.
* Complete figure-generation pipeline: 10 figures + 3 tables reproducible
  from checkpoints in under 1 hour on RTX 4090.

Reviewers wishing to verify the paper's headline numbers should run
`train_all.py --publication --seeds 0 1 2` and then rerun all figure scripts.
This takes approximately 24 hours of RTX 4090 compute.

## Known limitations of the reference implementation

Documented in `IMPLEMENTATION_PLAN.md`:

1. **P_max sensitivity**: extra CFAR false-positive candidates hurt receiver
   convergence more than the SBL baseline (which prunes via ARD). The
   reference figures use `P_max=P` (perfect cardinality assumption); a
   cardinality-agnostic receiver is future work.
2. **Safeguarded LM support step is not end-to-end differentiable** by
   architectural design (discrete accept/reject). `gamma_lm` remains a
   manually-tuned hyperparameter.
3. **beta_raw of layer 0** receives no gradient because `z_prev=None` at the
   first layer (mathematically correct but visible in gradient-flow tests).
4. **Wall-clock timing on RTX 4090** is dominated by the CG-MMSE step; the
   Set-Transformer and safeguarded LM add < 5% overhead each.

## Contacting the author

Any reproducibility issues, please contact the author with:
- Exact command used
- Full log output (redirect to file with `2>&1 > log`)
- GPU model and driver version
- PyTorch version

---

## Added 2026-08-06

### New scripts

| script | output | purpose |
|---|---|---|
| `crb_nuisance.py` | `runs/crb_nuisance.json` | joint 4P FIM + Schur complement; nuisance-eliminated CRB (Table V) |
| `highorder_sweep.py` | `runs/highorder_sweep.json` | 16-QAM vs QPSK SNR sweep with genie reference |
| `convergence_v3.py` | `runs/convergence_v3.json` | outer-loop convergence from the **actual** calibrated receiver |
| `aperture_ablation.py` | `runs/aperture_ablation.json` | coherent κ-refinement on/off at fixed everything else |
| `dgesbl_baseline.py` | — | D-GESBL-style adapted baseline (library) |
| `run_dgesbl_easy.py` / `run_dgesbl_hard.py` | `runs/dgesbl_easy.json`, `runs/dgesbl_hard.json` | baseline at the paper protocol |

### Notes

- **`convergence_v2.py` is superseded.** It duplicated the receiver inline and predated the `v_eff` calibration, so its numbers no longer match Table II. Use `convergence_v3.py`.
- **`make_paper_plots.py` holds stale values** for Figs. 2 and 3 (basin 0.0055/0.0096 vs the paper's 0.0059/0.0099). The paper's figures come from `audit_invariants.py` (Tests 2/2b) and `theta_sensitivity.py`; both reproduce exactly on re-run but write no JSON. **Archive their output before any further edits.**
- **`dgesbl_baseline.py` must be included in the public release** — a claim about a competing method rests on it. It is an *adaptation*: MMV with shared support and free per-block gains, embedded rather than superimposed pilots, no GAMP variant. Best config `T_em=40, grid_lr=0.1` from a 20-point sweep; untuned it scores 63% on Hard B=1 versus 43.8% tuned.
- **Receiver defaults are unchanged** (`soft_symbols=False`, `calibrate_output=False`), so every published QPSK number reproduces. The soft-symbol path is used only for the higher-order-modulation study.
- **Operator verification**: the fast operator agrees with a dense `F_A Δ_κ Π_ℓ F_A^H` reference to 2.7e-05 (float32 roundoff, no bias over 5 draws); the reference satisfies Φ(0,0)=I to 3e-15 and unitarity to 4e-15. Worth keeping as a regression test.
- **Long runs must save incrementally.** `run_dgesbl_eval.py` died after B=4 and lost everything; the replacements dump JSON after each B.

### Hyperparameter robustness sweep (Table V claim)

`scripts/hp_robustness.py` -> `runs/hp_robustness.json`, `runs/hp_robustness.log`.

Scales each of the 10 receiver hyperparameters in turn to 0.7x / 1.3x its
default (integers rounded) at (P=5, N_p=32), B=4, 15 dB, holding all others
fixed. Seeds are `k*137+42, k=0..4`, matching `ablation_v2.py`, so the
unperturbed run reproduces the Table III "Full MB-IDAR" entry (12.98%) exactly
and doubles as a fidelity check.

Result: max |delta| = 0.55 pp over 20 perturbations. Largest sensitivities are
kappa_window (+0.55), max_step (-0.53) and rho_min (+0.51); lambda_ridge, slack
and K_cg are numerically inert.

Note: this script reports sample std (ddof=1); the published tables use ddof=0.
Means -- and therefore all reported deltas -- are unaffected.

To enable the sweep, `multiblock_dasbl_receiver` gained pass-through arguments
`gamma_lr, max_step, slack, kappa_window, kappa_step, K_cg`, whose defaults are
exactly the previously hardcoded values. All published numbers are unchanged.

### D-GESBL-style baseline re-tuning (2026-08-06)

The original tuning sweep script was lost; only `runs/dgesbl_best_config.json`
(the winner) survived, so the Table I caption's "reported at its best" was not
reproducible. Reconstructed as:

- `scripts/tune_dgesbl.py`     -> 20-point grid, T_em x grid_lr
- `scripts/tune_dgesbl_ext.py` -> extends to 32 points (T_em up to 160)
- `runs/dgesbl_tuning.json` / `.log`

The 20-point grid's optimum sat at T_em=40, its LARGEST value -- a boundary
optimum, i.e. the baseline was under-tuned. Extending T_em to 160 improves it
by ~2 pp at every B; grid_lr=0.1 stays optimal (interior) throughout. New
optimum: T_em=160, grid_lr=0.1.

`scripts/dgesbl_Tem_check.py` / `runs/dgesbl_Tem_check.json` confirm the
saturation in B is structural, not an artifact of too few EM iterations:

    B      T_em=40   T_em=160
    1       44.71      42.14
    2       34.79      32.22
    4       29.75      28.01
    8       30.35      28.39      (flat past B=4 at both settings)

`scripts/run_dgesbl_retuned.py` -> `runs/dgesbl_retuned.json` re-runs every
reported baseline number at the new optimum under the full 5x8x32 protocol.
This SUPERSEDES dgesbl_{easy,hard,fairpilots}.json in Tables I and II.

Effect on the manuscript: Hard B=1 43.8 -> 40.6%, Easy B=1 8.4 -> 8.2%,
fair-pilot P=5 7.9 -> 7.7%. The advantage ratios fall from 1.27/1.69/1.89x to
1.16/1.58/1.77x, so the headline is now 1.8x, and "saturates past B=2" became
"saturates past B=4". The regime-scoping conclusion is unchanged.

### Coarse-support RMSE (Fig. 3 shaded band)

`scripts/coarse_rmse.py` -> `runs/coarse_rmse.json`. Measures the peak-selection
+ 2-Newton front end from pilot-only initialization at 15 dB, matching estimated
to true paths by optimal assignment on (ell, kappa):

                     detected-only    all paths    detection rate
    Easy (P=3,Np=32)     0.240          1.213           70%
    Hard (P=5,Np=16)     0.299          1.563           42%

"detected" = within 0.75 of a true path in both coordinates. The manuscript
previously claimed 0.3-0.5 "at the basin edge on Hard and past it on Easy" with
no supporting artifact; the ordering was also backwards (Hard is worse). Both
the text and the Fig. 3 band now use the measured values. The all-paths figure
is consistent with the B=1 per-path RMSE in Table IV (1.55 / 1.48).

### Known unverifiable claim

The sentence "under the nominal weighting the same sweep spanned 23.4% to 18.2%"
(noise-mismatch robustness) has no surviving run file; `runs/robustness_v2.json`
records only the calibrated arm. Re-run or drop before camera-ready.
