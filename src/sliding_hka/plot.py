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
    max_gap_bp: int = 2000,
) -> Figure:
    """Render observed vs. expected silent pairwise diversity along a CDS.

    When the sliding-window output contains stretches of NaN (alignment
    dead zones) at least ``max_gap_bp`` long, those stretches are removed
    from the x-axis and the plot is split into multiple panels with a
    broken-axis indicator between them. Shorter NaN gaps appear as breaks
    in the line within a panel. Set ``max_gap_bp`` very large to force a
    single-panel layout.
    """
    sns.set_theme(context="paper", style="white")

    pos = np.asarray(window_output["nt_position"])
    obs = np.asarray(window_output["obs_pi"])
    exp = np.asarray(window_output["exp_pi"])

    segments = _segments_from_gaps(pos, obs, max_gap_bp)
    n_seg = len(segments)

    if annotation is not None:
        fig = plt.figure(
            figsize=(max(8, 1.5 * n_seg + 6), 4.2), constrained_layout=True
        )
        widths = [hi - lo + 1 for lo, hi in segments]
        gs = fig.add_gridspec(
            2, n_seg, height_ratios=[6, 1], width_ratios=widths, wspace=0.05
        )
        main_axes = [fig.add_subplot(gs[0, i]) for i in range(n_seg)]
        track_axes = [
            fig.add_subplot(gs[1, i], sharex=main_axes[i]) for i in range(n_seg)
        ]
    else:
        fig = plt.figure(
            figsize=(max(8, 1.5 * n_seg + 6), 3.5), constrained_layout=True
        )
        widths = [hi - lo + 1 for lo, hi in segments]
        gs = fig.add_gridspec(1, n_seg, width_ratios=widths, wspace=0.05)
        main_axes = [fig.add_subplot(gs[0, i]) for i in range(n_seg)]
        track_axes = None

    # Share y-axis across main panels
    for ax in main_axes[1:]:
        ax.sharey(main_axes[0])

    for i, (lo, hi) in enumerate(segments):
        ax = main_axes[i]
        mask = (pos >= lo) & (pos <= hi)
        ax.plot(pos[mask], obs[mask], color="black", lw=1.5,
                label="Observed pairwise π (silent)" if i == 0 else None)
        ax.plot(pos[mask], exp[mask], color="grey", lw=1.5, ls="--",
                label="Expected from divergence / (T+1)" if i == 0 else None)
        ax.set_xlim(lo, hi)
        ax.spines["top"].set_visible(False)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
        if i < n_seg - 1:
            ax.spines["right"].set_visible(False)

    main_axes[0].set_ylabel("Silent pairwise differences per site")
    main_axes[0].legend(frameon=False, fontsize="small", loc="best")

    # Break markers between adjacent main panels
    for i in range(n_seg - 1):
        _draw_break_marker(main_axes[i], main_axes[i + 1])

    # Gene-structure track per segment
    if track_axes is not None:
        for i, (lo, hi) in enumerate(segments):
            _draw_gene_structure(track_axes[i], annotation, xlim=(lo, hi))
            track_axes[i].spines["top"].set_visible(False)
            track_axes[i].spines["left"].set_visible(False)
            track_axes[i].spines["right"].set_visible(False)
            if i > 0:
                track_axes[i].tick_params(left=False, labelleft=False)
            if i > 0:
                track_axes[i].spines["left"].set_visible(False)
            if i < n_seg - 1:
                track_axes[i].spines["right"].set_visible(False)
        for i in range(n_seg - 1):
            _draw_break_marker(track_axes[i], track_axes[i + 1])
        for ax in main_axes:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")
        # Single x-label on the figure (centered below all track panels)
        fig.supxlabel("Nucleotide position (bp)")
    else:
        fig.supxlabel("Nucleotide position (bp)")

    fig.suptitle(f"{locus}  —  T+1={t_plus_1:.2f}, w={window} silent sites")

    if save_to is not None:
        fig.savefig(save_to, dpi=150)

    return fig


def _segments_from_gaps(
    pos: np.ndarray, obs: np.ndarray, max_gap_bp: int
) -> list[tuple[int, int]]:
    """Return [(lo, hi), ...] bp ranges of segments separated by long NaN gaps.

    NaN runs whose bp span is >= ``max_gap_bp`` are treated as cuts; their
    interior is excluded from the plot. Shorter runs are kept inside a
    segment (rendered as line gaps).
    """
    if len(pos) == 0:
        return [(0, 0)]

    is_missing = np.isnan(obs)
    if not is_missing.any():
        return [(int(pos[0]), int(pos[-1]))]

    # Contiguous NaN runs (index-space)
    edges = np.diff(is_missing.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if is_missing[0]:
        starts = np.r_[0, starts]
    if is_missing[-1]:
        ends = np.r_[ends, len(is_missing)]

    long_cut_bp_ranges: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        bp_lo = int(pos[s])
        bp_hi = int(pos[e - 1])
        if bp_hi - bp_lo + 1 >= max_gap_bp:
            long_cut_bp_ranges.append((bp_lo, bp_hi))

    if not long_cut_bp_ranges:
        return [(int(pos[0]), int(pos[-1]))]

    # Segments between cuts
    segments: list[tuple[int, int]] = []
    seg_lo = int(pos[0])
    for cut_lo, cut_hi in sorted(long_cut_bp_ranges):
        if cut_lo > seg_lo:
            segments.append((seg_lo, cut_lo - 1))
        seg_lo = cut_hi + 1
    if seg_lo <= int(pos[-1]):
        segments.append((seg_lo, int(pos[-1])))

    # Guard against empty list (all-NaN input)
    if not segments:
        segments = [(int(pos[0]), int(pos[-1]))]
    return segments


def _draw_break_marker(ax_left, ax_right) -> None:
    """Draw matched diagonal break markers between two adjacent panels."""
    d = 0.012
    kwargs = dict(color="k", clip_on=False, lw=1, solid_capstyle="butt")
    ax_left.plot(
        [1 - d, 1 + d], [-d, +d], transform=ax_left.transAxes, **kwargs
    )
    ax_left.plot(
        [1 - d, 1 + d], [1 - d, 1 + d], transform=ax_left.transAxes, **kwargs
    )
    ax_right.plot(
        [-d, +d], [-d, +d], transform=ax_right.transAxes, **kwargs
    )
    ax_right.plot(
        [-d, +d], [1 - d, 1 + d], transform=ax_right.transAxes, **kwargs
    )


def _draw_gene_structure(
    ax, annotation: LocusAnnotation, xlim: tuple[int, int] | None = None
) -> None:
    """Draw exon/UTR/intron structure on the given axes.

    * CDS        -> filled black rectangle (thick)
    * UTR        -> open outlined rectangle (thin)
    * intron     -> thin black baseline rectangle
    * intergenic -> blank
    """
    feat = annotation.feature
    n = annotation.n_positions

    ax.axhline(0.0, color="black", lw=0.5, zorder=1)

    cds_runs = _contiguous_runs(feat, "CDS")
    utr_runs = (
        _contiguous_runs(feat, "five_prime_UTR")
        + _contiguous_runs(feat, "three_prime_UTR")
    )
    intron_runs = _contiguous_runs(feat, "intron")

    for s, e in intron_runs:
        ax.add_patch(Rectangle((s + 1, -0.02), e - s, 0.04,
                               facecolor="black", edgecolor="none", zorder=2))
    for s, e in utr_runs:
        ax.add_patch(Rectangle((s + 1, -0.2), e - s, 0.4,
                               facecolor="none", edgecolor="black", lw=1.0,
                               zorder=3))
    for s, e in cds_runs:
        ax.add_patch(Rectangle((s + 1, -0.4), e - s, 0.8,
                               facecolor="black", edgecolor="black", zorder=4))

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(1, n)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])


def _contiguous_runs(feat: np.ndarray, label: str) -> list[tuple[int, int]]:
    """Return [(start, end)] half-open 0-based index runs of ``label``."""
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
