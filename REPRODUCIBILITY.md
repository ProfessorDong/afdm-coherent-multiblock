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
