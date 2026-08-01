"""Shared plotting utilities for all P4 figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt


PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def set_paper_style():
    for k, v in PAPER_STYLE.items():
        matplotlib.rcParams[k] = v


COLORS = {
    "genie": "black",
    "classical": "tab:blue",
    "pbigabp": "tab:green",
    "jpnce_sbl": "tab:orange",
    "proposed": "tab:red",
    "ungated": "tab:purple",
    "scalars": "tab:cyan",
    "attention": "tab:pink",
    "gate": "tab:brown",
}

MARKERS = {
    "genie": "*",
    "classical": "s",
    "pbigabp": "^",
    "jpnce_sbl": "v",
    "proposed": "o",
    "ungated": "D",
    "scalars": "P",
    "attention": "X",
    "gate": "H",
}

LABELS = {
    "genie": "Perfect-CSI CG-MMSE",
    "classical": "Classical semi-blind CG",
    "pbigabp": "PBiGaBP [Ranasinghe et al. 2025]",
    "jpnce_sbl": "JPNCE-SBL [Xu et al. 2026]",
    "proposed": "Proposed (V-EM + gate + attn)",
    "ungated": "Ungated V-EM (ablation)",
    "scalars": "Learned scalars only",
    "attention": "+ Set-Transformer (no gate)",
    "gate": "+ Uncertainty gate (no attn)",
}


def figures_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig, name: str) -> Path:
    """Save figure as PDF (paper-quality) and PNG (quick-view)."""
    outdir = figures_dir()
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf
