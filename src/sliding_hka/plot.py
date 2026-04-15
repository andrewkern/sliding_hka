"""Fig 5-style sliding-window HKA diagnostic plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def sliding_hka_plot(
    window_output: dict[str, np.ndarray],
    t_plus_1: float,
    window: int,
    locus: str,
    save_to: str | Path | None = None,
) -> Figure:
    """Render observed vs. expected silent pairwise diversity along a CDS.

    Args:
        window_output: Dict from ``sliding_hka.window.sliding_window``.
        t_plus_1: Divergence-time scaler used to compute ``exp_pi``.
        window: Window width in silent sites (for the title).
        locus: Locus name (for the title).
        save_to: If provided, also save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    sns.set_theme(context="paper", style="white")

    pos = window_output["nt_position"]
    obs = window_output["obs_pi"]
    exp = window_output["exp_pi"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(pos, obs, color="black", lw=1.5, label="Observed pairwise π (silent)")
    ax.plot(
        pos,
        exp,
        color="grey",
        lw=1.5,
        ls="--",
        label="Expected from divergence / (T+1)",
    )

    ax.set_xlabel("Nucleotide position (bp)")
    ax.set_ylabel("Silent pairwise differences per site")
    ax.set_title(f"{locus}  —  T+1={t_plus_1:.2f}, w={window} silent sites")
    ax.legend(frameon=False, fontsize="small", loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=150)

    return fig
