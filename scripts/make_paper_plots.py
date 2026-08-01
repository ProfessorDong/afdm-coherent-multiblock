"""Generate publication-quality figures for the paper.

Produces PDF figures with matplotlib:
  Fig 1: Pilot bias floor + basin of attraction (dual panel)
  Fig 2: BER vs SNR for easy config, all methods
  Fig 3: BER vs SNR for hard config, all methods
  Fig 4: Scaling with number of blocks B
  Fig 5: Phase diagram heatmap
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.figsize": (5.5, 4.0),
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
})


OUT_DIR = Path("paper/figures_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_ber_vs_snr():
    """Fig 2 + 3: BER vs SNR for easy and hard configs."""
    for cfg_tag, (P, Np, name) in {
        "easy": (3, 32, "EASY: P=3, N_p=32"),
        "hard": (5, 16, "HARD: P=5, N_p=16"),
    }.items():
        with open(f"runs/ber_vs_snr_{P}_{Np}.json") as f:
            data = json.load(f)

        fig, ax = plt.subplots()
        snrs = data["snrs"]
        methods = [
            ("genie", "Genie MMSE", "black", "-", "o"),
            ("classical", "Classical CG", "gray", "--", "s"),
            ("sb_dasbl", "SB-DASBL (ours)", "tab:orange", "-", "D"),
            ("mb_b2", "MB-DASBL B=2 (ours)", "tab:blue", "-", "^"),
            ("mb_b4", "MB-DASBL B=4 (ours)", "tab:red", "-", "v"),
            ("oracle_theta", "Oracle-θ DASBL", "black", ":", "*"),
        ]
        for key, label, color, ls, mk in methods:
            ax.semilogy(snrs, data[key], label=label, color=color, linestyle=ls, marker=mk)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Symbol Error Rate")
        ax.set_title(name)
        ax.set_ylim(1e-4, 1)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = OUT_DIR / f"fig_ber_vs_snr_{cfg_tag}.pdf"
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")


def fig_scaling_with_B():
    """Fig 4: SER scaling with number of blocks B (log-log)."""
    # From the multi-block scaling experiments
    data = {
        "Easy (P=3, N_p=32)":   {"B": [1, 2, 4, 8], "SER": [0.121, 0.042, 0.026, 0.021]},
        "Hard (P=5, N_p=16)":   {"B": [1, 2, 4, 8], "SER": [0.577, 0.281, 0.145, 0.085]},
        "Hard (P=5, N_p=32)":   {"B": [1, 2, 4, 8], "SER": [0.315, 0.118, 0.079, 0.049]},
    }
    fig, ax = plt.subplots()
    for label, d in data.items():
        ax.loglog(d["B"], d["SER"], marker="o", label=label)
    # Ideal 1/B reference
    Br = np.array([1, 2, 4, 8])
    for label, d in data.items():
        ref = d["SER"][0] / Br
        ax.loglog(Br, ref, linestyle=":", alpha=0.4, color="gray")
    ax.set_xlabel("Number of Blocks B")
    ax.set_ylabel("SER at 15 dB")
    ax.set_title("Multi-Block DASBL Scaling")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "fig_scaling_B.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig_basin_of_attraction():
    """Fig 1b: Basin of attraction of iterative DASBL."""
    # From theta_sensitivity.py output at SNR=15dB
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]
    ser_hard = [0.0096, 0.0129, 0.0202, 0.0859, 0.2476, 0.4688, 0.5233, 0.6184, 0.6701, 0.6934]
    ser_easy = [0.0055, 0.0060, 0.0090, 0.0369, 0.1113, 0.3774, 0.4453, 0.5813, 0.6631, 0.6573]

    fig, ax = plt.subplots()
    ax.plot(sigmas, ser_easy, marker="o", label="Easy (P=3, N_p=32)", color="tab:blue")
    ax.plot(sigmas, ser_hard, marker="s", label="Hard (P=5, N_p=16)", color="tab:red")
    ax.axvline(0.2, linestyle=":", color="gray", alpha=0.7, label=r"Basin edge $\sigma \approx 0.2$")
    ax.axvspan(0.3, 0.5, alpha=0.1, color="gray", label="Typical CFAR RMSE")
    ax.set_xlabel(r"Initial $(\ell, \kappa)$ perturbation std. $\sigma$")
    ax.set_ylabel("Final SER (SNR = 15 dB)")
    ax.set_title(r"Basin of Attraction of Iterative Data-Aided SBL")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "fig_basin.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig_pilot_bias_floor():
    """Fig 1a: Pilot bias floor from audit Test 2b."""
    # From audit_invariants.py Test 2 (true x) and Test 2b (pilot-only x)
    snrs = [5, 15, 25, 35, 50]
    nmse_truex = [4.478e-3, 4.478e-4, 4.478e-5, 4.478e-6, 1.417e-7]
    nmse_pilotonly = [8.159e-2, 6.544e-2, 6.335e-2, 6.298e-2, 6.289e-2]

    fig, ax = plt.subplots()
    ax.semilogy(snrs, nmse_pilotonly, marker="s", label="Pilot-only regression (biased)",
                color="tab:red")
    ax.semilogy(snrs, nmse_truex, marker="o", label="Data-aided regression (unbiased)",
                color="tab:blue")
    ax.axhline(6.3e-2, linestyle=":", color="gray", label=r"Bias floor $\approx 6.3\%$ NMSE")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"NMSE of $\hat{h}$")
    ax.set_title("Pilot-Only Regression Bias Floor (P=3, N_p=32)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "fig_bias_floor.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig_phase_diagram():
    """Fig 5: Phase diagram heatmap."""
    # SER of our single-block receiver at SNR=15dB across (P, N_p)
    # From phase_diagram.py
    P_vals = [2, 3, 5, 7]
    Np_vals = [8, 16, 24, 32, 48]
    receiver_ser = np.array([
        [0.557, 0.231, 0.103, 0.090, 0.036],  # P=2
        [0.626, 0.366, 0.202, 0.160, 0.074],  # P=3
        [0.671, 0.508, 0.394, 0.306, 0.174],  # P=5
        [0.684, 0.597, 0.459, 0.375, 0.235],  # P=7
    ])
    oracle_theta = np.array([
        [0.008, 0.005, 0.005, 0.004, 0.004],
        [0.011, 0.004, 0.003, 0.005, 0.004],
        [0.024, 0.009, 0.005, 0.006, 0.006],
        [0.061, 0.008, 0.007, 0.007, 0.004],
    ])

    gap = np.log10(receiver_ser + 1e-6) - np.log10(oracle_theta + 1e-6)  # log-gap

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(gap, cmap="YlOrRd", origin="lower", aspect="auto",
                   vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(Np_vals)))
    ax.set_xticklabels(Np_vals)
    ax.set_yticks(np.arange(len(P_vals)))
    ax.set_yticklabels(P_vals)
    ax.set_xlabel(r"Pilot count $N_p$")
    ax.set_ylabel(r"Path count $P$")
    ax.set_title(r"Identifiability Gap: $\log_{10}(\mathrm{SER}_{rx} / \mathrm{SER}_{oracle-\theta})$")
    for i in range(len(P_vals)):
        for j in range(len(Np_vals)):
            ax.text(j, i, f"{gap[i,j]:.1f}", ha="center", va="center",
                    color="black" if gap[i,j] < 1.5 else "white", fontsize=8)
    cbar = plt.colorbar(im)
    cbar.set_label("log gap (decades)")
    plt.tight_layout()
    path = OUT_DIR / "fig_phase_diagram.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig_convergence():
    """Fig: SER vs outer iteration for MB-DASBL B=4."""
    import json
    with open("runs/convergence_trace.json") as f:
        data = json.load(f)
    fig, ax = plt.subplots()
    for label, traj in data.items():
        iters = list(range(len(traj)))
        ax.semilogy(iters, traj, marker="o", label=label)
    ax.set_xlabel("Outer iteration $t$")
    ax.set_ylabel("SER at 15 dB")
    ax.set_title("Multi-Block DASBL Convergence ($B=4$)")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "fig_convergence.pdf"
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")


def fig_tdlc():
    """Fig: TDL-C SER vs B for two Doppler values."""
    data = {
        "TDL-C $P_{use}=5$, $\\nu_{max}=500$ Hz":    [1.47, 1.23, 1.16, 1.24],
        "TDL-C $P_{use}=5$, $\\nu_{max}=3000$ Hz":   [3.19, 2.17, 2.48, 2.46],
        "TDL-C $P_{use}=7$, $\\nu_{max}=500$ Hz":    [1.79, 1.37, 1.24, 1.31],
        "TDL-C $P_{use}=7$, $\\nu_{max}=3000$ Hz":   [3.80, 2.21, 2.38, 2.75],
    }
    Bs = [1, 2, 4, 8]
    fig, ax = plt.subplots()
    for label, y in data.items():
        ax.semilogy(Bs, [v/100.0 for v in y], marker="o", label=label)
    ax.set_xlabel("Number of Blocks $B$")
    ax.set_ylabel("SER at 15 dB")
    ax.set_title("TDL-C Channel Model, MB-DASBL")
    ax.set_xticks(Bs)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "fig_tdlc.pdf"
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")


def fig_baselines_comparison():
    """Bar chart: comparison against JPNCE-SBL, PBiGaBP, learned Set-Transformer."""
    methods = ["Classical\nCG", "JPNCE-SBL\n(Xu 2026)", "PBiGaBP\n(2024)",
               "Learned\nSet-Transformer", "SB-DASBL\n(ours)", "MB-DASBL\nB=8 (ours)",
               "Genie\nMMSE"]
    easy_ser = [0.292, 0.268, 0.289, 0.162, 0.164, 0.026, 0.005]
    hard_ser = [0.519, 0.504, 0.517, 0.450, 0.505, 0.090, 0.005]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    b1 = ax.bar(x - w/2, easy_ser, w, label="Easy (P=3, N_p=32)", color="tab:blue")
    b2 = ax.bar(x + w/2, hard_ser, w, label="Hard (P=5, N_p=16)", color="tab:red")
    ax.set_yscale("log")
    ax.set_ylabel("SER at 15 dB")
    ax.set_title("Receiver Comparison at 15 dB")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.legend(loc="lower left")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    # Annotations
    for bars, vals in [(b1, easy_ser), (b2, hard_ser)]:
        for b, v in zip(bars, vals):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h * 1.15, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = OUT_DIR / "fig_comparison_bar.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("Generating paper figures...")
    fig_pilot_bias_floor()
    fig_basin_of_attraction()
    fig_ber_vs_snr()
    fig_scaling_with_B()
    fig_phase_diagram()
    fig_convergence()
    fig_tdlc()
    fig_baselines_comparison()
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
