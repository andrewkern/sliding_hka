"""Fig 5-style sliding-window HKA diagnostic plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from sliding_hka.annotation import LocusAnnotation


def sliding_hka_plot(
    window_output: dict[str, np.ndarray],
    t_plus_1: float,
    window: int,
    locus: str,
    save_to: str | Path | None = None,
    annotation: LocusAnnotation | None = None,
) -> Figure:
    """Render observed vs. expected silent pairwise diversity along a CDS.

    Args:
        window_output: Dict from ``sliding_hka.window.sliding_window``.
        t_plus_1: Divergence-time scaler used to compute ``exp_pi``.
        window: Window width in silent sites (for the title).
        locus: Locus name (for the title).
        save_to: If provided, also save the figure to this path.
        annotation: If provided, render a gene-structure track below the
            main plot (CDS filled blocks, UTR open blocks, introns as a
            thin line), matching the Kreitman & Hudson 1991 Fig 5 style.

    Returns:
        The matplotlib Figure.
    """
    sns.set_theme(context="paper", style="white")

    pos = np.asarray(window_output["nt_position"])
    obs = np.asarray(window_output["obs_pi"])
    exp = np.asarray(window_output["exp_pi"])

    if annotation is not None:
        fig = plt.figure(figsize=(9, 4.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        track_ax = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        track_ax = None

    ax.plot(pos, obs, color="black", lw=1.5, label="Observed pairwise π (silent)")
    ax.plot(
        pos,
        exp,
        color="grey",
        lw=1.5,
        ls="--",
        label="Expected from divergence / (T+1)",
    )

    _shade_missing_regions(ax, pos, obs)

    ax.set_ylabel("Silent pairwise differences per site")
    ax.set_title(f"{locus}  —  T+1={t_plus_1:.2f}, w={window} silent sites")
    ax.legend(frameon=False, fontsize="small", loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if track_ax is not None:
        _draw_gene_structure(track_ax, annotation)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)
        track_ax.set_xlabel("Nucleotide position (bp)")
    else:
        ax.set_xlabel("Nucleotide position (bp)")

    if annotation is None:
        fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=150)

    return fig


def _shade_missing_regions(ax, pos: np.ndarray, obs: np.ndarray) -> None:
    """Shade contiguous NaN-obs regions (alignment dead zones) in light grey."""
    if len(obs) == 0:
        return
    is_missing = np.isnan(obs)
    if not is_missing.any():
        return
    # Find contiguous runs of True in is_missing
    edges = np.diff(is_missing.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if is_missing[0]:
        starts = np.r_[0, starts]
    if is_missing[-1]:
        ends = np.r_[ends, len(is_missing)]
    for s, e in zip(starts, ends):
        if e - s < 2:  # skip single-point gaps
            continue
        ax.axvspan(pos[s], pos[e - 1], facecolor="0.90", edgecolor="none",
                   zorder=0, alpha=1.0)


def _draw_gene_structure(ax, annotation: LocusAnnotation) -> None:
    """Draw exon/UTR/intron structure on the given axes.

    Following the 1991 paper's aesthetic:
      * CDS      -> filled black rectangle (thick)
      * UTR      -> open rectangle (thin)
      * intron   -> horizontal line (zero height)
      * intergenic -> nothing
    """
    feat = annotation.feature
    n = annotation.n_positions

    # Draw a baseline so the gene body reads as a single object
    ax.axhline(0.0, color="black", lw=0.5, zorder=1)

    # Collect contiguous runs per feature
    cds_runs = _contiguous_runs(feat, "CDS")
    utr_runs = (
        _contiguous_runs(feat, "five_prime_UTR")
        + _contiguous_runs(feat, "three_prime_UTR")
    )
    intron_runs = _contiguous_runs(feat, "intron")

    # Introns as a thin baseline (already drawn as axhline, but reinforce)
    for s, e in intron_runs:
        ax.add_patch(Rectangle((s + 1, -0.02), e - s, 0.04,
                               facecolor="black", edgecolor="none", zorder=2))

    # UTRs as a thinner open block
    for s, e in utr_runs:
        ax.add_patch(Rectangle((s + 1, -0.2), e - s, 0.4,
                               facecolor="none", edgecolor="black", lw=1.0,
                               zorder=3))

    # CDS as thick filled block
    for s, e in cds_runs:
        ax.add_patch(Rectangle((s + 1, -0.4), e - s, 0.8,
                               facecolor="black", edgecolor="black", zorder=4))

    ax.set_xlim(1, n)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _contiguous_runs(feat: np.ndarray, label: str) -> list[tuple[int, int]]:
    """Return list of (start, end) half-open 0-based index runs of ``label``."""
    mask = feat == label
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return list(zip(starts.tolist(), ends.tolist()))
