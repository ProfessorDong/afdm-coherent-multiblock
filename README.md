# AFDM Multi-Block Iterative Data-Aided Receiver (MB-IDAR)

Reference implementation and experimental artifacts for

> **Iterative Multi-Block Data-Aided Off-Grid Channel Estimation and Detection for AFDM**
> *L. Dong*, IEEE Trans. Commun. (under review), 2026.

The receiver alternates reliability-weighted data-aided regression, safeguarded gradient support refinement, and CG-MMSE detection over a coherence window of `B` AFDM blocks sharing the same physical channel but using per-block pilot diversity.

## What's here

```
afdm/                Core library (system, operators, channels, classical & SBL & BP baselines)
scripts/             Experiment scripts that reproduce the figures/tables in the paper
tests/               Unit tests for the core primitives
runs/                Numerical results (JSON) underlying the paper's plots
requirements.txt     Python dependencies
```

Selected scripts:

| Script | Reproduces |
|---|---|
| `scripts/ber_vs_snr.py` | Fig 4 / Fig 5 (SER vs SNR, Easy / Hard) |
| `scripts/multiblock_dasbl.py` | Fig 6 (SER scaling in `B`), Table II |
| `scripts/multi_seed_error_bars.py` | Table I (SER at 15 dB with error bars) |
| `scripts/channel_aging.py` | Fig 10 (robustness to shared-support violation) |
| `scripts/convergence_trace.py` | Fig 8 (outer-loop convergence) |
| `scripts/phase_diagram.py` | Fig 7 (empirical recovery-regime map) |
| `scripts/tdlc_evaluation.py` | Fig 9 (3GPP TDL-C evaluation) |
| `scripts/run_ablation.py` | Table III (ablation) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce Fig 6 (SER vs B at 15 dB)
python scripts/multiblock_dasbl.py --Bs 1,2,4,8 --configs all

# Reproduce Fig 10 (channel aging)
python scripts/channel_aging.py
```

Results are written under `runs/`; JSON files in `runs/` reflect the exact numbers used in the paper.

## Reproducing the paper's numbers

- All headline SER numbers use 10 seeds × 8 batches × 32 realizations (see `multi_seed_error_bars.py`).
- Fig 10 (channel aging) uses 5 seeds × 4 batches × 16 realizations.
- See `REPRODUCIBILITY.md` for the full protocol.

## Notes

- **No trained weights ship with this repo.** All algorithms are classical/analytical: CFAR + Newton support recovery, ridge LS gain estimation, CG-MMSE symbol detection, reliability-weighted pseudo-pilots, safeguarded-gradient support refinement, and multi-block variants of the above. Anything you might find in a companion repository under `checkpoints/` belongs to an earlier learned-receiver project and is unrelated to this paper.
- GPU acceleration is optional; scripts run on CPU (slower).

## Citation

```bibtex
@article{dong_afdm_mbidar_2026,
  author  = {Dong, Liang},
  title   = {Iterative Multi-Block Data-Aided Off-Grid Channel Estimation
             and Detection for AFDM},
  journal = {IEEE Trans. Commun.},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see `LICENSE`.
