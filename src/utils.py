"""
utils.py
--------
Shared plotting helpers and formatting utilities.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "accent":    "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#6B7280",
    "bg":        "#F9FAFB",
}

SEGMENT_COLORS = {
    "Champions":           "#10B981",
    "Loyal Customers":     "#059669",
    "Potential Loyalists": "#34D399",
    "Recent Customers":    "#3B82F6",
    "At-Risk":             "#F59E0B",
    "Cant Lose Them":      "#EF4444",
    "Hibernating":         "#9CA3AF",
    "Lost":                "#6B7280",
}

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")


def apply_style() -> None:
    """Apply a clean, report-ready matplotlib style."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.4,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })


def save_fig(fig: plt.Figure, name: str, dpi: int = 150) -> str:
    """Save a figure to outputs/figures/ and return the path."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fmt_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"
