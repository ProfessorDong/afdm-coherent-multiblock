# AFDM Coherent Multi-Block Doppler Estimation (MB-IDAR)

Reference implementation and experimental artifacts for

> **Coherent Multi-Block Doppler Estimation and Data-Aided Detection for AFDM**
> *L. Dong*, IEEE Trans. Commun. (under review), 2026.

Consecutive AFDM blocks accrue a deterministic inter-block Doppler phase
`D_b(kappa) = diag(exp(j 2 pi kappa_p b beta))`, `beta = (N + N_cp)/N`, so `B`
contiguous blocks form a **slow-time aperture** whose Doppler Fisher information
grows as `Theta(B^3)` rather than the `Theta(B)` of independent averaging. The
same ramp is periodic, creating a Doppler-Nyquist ambiguity of period `1/beta`,
which forces a coarse-to-fine design: non-coherent peak acquisition, a
sub-Nyquist coherent local search (*aperture synthesis*), and a phase-corrected
data-aided loop calibrated by a measured effective variance `v_eff`.

## What's here

```
afdm/                Core library (system, operators, channels, classical/SBL/BP baselines)
scripts/             Experiment scripts reproducing the paper's figures and tables
tests/               Unit tests for the core primitives
runs/                Numerical results (JSON) underlying every reported number
requirements.txt     Python dependencies
```

## Reproducing the paper

Every number in the paper traces to a JSON file under `runs/`.

| Paper item | Script | Artifact |
|---|---|---|
| Table I (SER at 15 dB) | `multi_seed_error_bars.py`, `scaling_B_v2.py`, `run_dgesbl_retuned.py` | `multiseed_15db.json`, `scaling_B_v2.json`, `dgesbl_retuned.json` |
| Table II (fair aggregate pilots) | `fair_pilots_v2.py`, `run_dgesbl_retuned.py` | `fair_pilots_v2.json`, `dgesbl_retuned.json` |
| Table III (ablation) | `ablation_v2.py` | `ablation_v2.json` |
| Table IV (nuisance-eliminated CRB) | `crb_nuisance.py`, `crb_vs_B.py` | `crb_nuisance.json`, `crb_vs_B.json` |
| Table V (hyperparameters) | `hp_robustness.py` (+/-30% sweep) | `hp_robustness.json` |
| Fig. 2 (pilot-only MSE floor) | `make_paper_plots.py` | inline in manuscript |
| Fig. 3 (basin of attraction) | `theta_sensitivity.py`, `coarse_rmse.py` | `coarse_rmse.json` |
| Figs. 4-5 (SER vs SNR) | `ber_vs_snr_v2.py` | `ber_vs_snr_v2_{3_32,5_16}.json` |
| Fig. 6 (SER scaling in B) | `scaling_B_v2.py` | `scaling_B_v2.json` |
| Fig. 7 (3GPP TDL-C) | `tdlc_evaluation.py` | `tdlc_v2.json` |
| Aperture ablation (Sec. V) | `aperture_ablation.py` | `aperture_ablation.json` |
| Cubic-bound attainability | `localml_bcubed.py` | `localml_bcubed.json` |
| 16-QAM / high-order study | `highorder_sweep.py` | `highorder_sweep.json` |
| Convergence in `T` | `convergence_v3.py` | `convergence_v3.json` |
| Noise-mismatch robustness | `robustness_v2.py` | `robustness_v2.json` |

See `REPRODUCIBILITY.md` for the exact protocol behind each entry, including
which files supersede earlier ones.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/scaling_B_v2.py        # Fig. 6 / Table I multi-block rows
python scripts/ablation_v2.py         # Table III
python scripts/crb_nuisance.py        # Table IV
```

Results are written under `runs/`. Seeds are `k*137 + 42` throughout, so reruns
reproduce the published values exactly.

## The D-GESBL-style baseline

`scripts/dgesbl_baseline.py` is an **adaptation** of Luo *et al.* (IEEE TCOM
2026, arXiv:2607.18881), not a reimplementation: the superimposed-pilot frame is
replaced by the embedded pilots every receiver here uses, so all methods see
identical observations, and the GAMP variant is omitted. It is reported at its
best over a 32-configuration sweep (`tune_dgesbl.py`, `tune_dgesbl_ext.py`).

The optimum is `T_em=160, grid_lr=0.1`. An earlier 20-point grid put it at
`T_em=40`, which was the grid's largest value -- a boundary optimum that left
the baseline under-tuned by about 2 pp at every `B`. `dgesbl_Tem_check.py`
confirms the baseline's saturation in `B` is structural rather than a shortage
of EM iterations: the `B=4` / `B=8` tie holds at both `T_em=40` and `T_em=160`.
This baseline beats our single-block receiver at both operating points; the
paper's claim is scoped accordingly.

## Notes

- **No trained weights ship with this repo.** Every algorithm here is
  classical/analytical: peak selection + Newton support recovery, ridge LS gain
  estimation, CG-MMSE detection, reliability-gated pseudo-pilots,
  safeguarded-gradient support refinement, and multi-block variants.
- `scripts/` also contains exploratory and diagnostic code from the development
  history that no paper result depends on; the table above lists what matters.
- GPU acceleration is optional; scripts run on CPU (slower).

## Citation

```bibtex
@article{dong_afdm_coherent_multiblock_2026,
  author  = {Dong, Liang},
  title   = {Coherent Multi-Block Doppler Estimation and Data-Aided
             Detection for AFDM},
  journal = {IEEE Trans. Commun.},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see `LICENSE`.
