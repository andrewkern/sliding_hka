"""Sliding window over per-codon silent-site / pi / divergence arrays."""

from __future__ import annotations

import numpy as np


def codon_nt_position(codon_index: int) -> int:
    """Return the (1-based) nucleotide midpoint of a codon index."""
    return 3 * codon_index + 1


def sliding_window(
    silent_sites: np.ndarray,
    silent_pi: np.ndarray,
    silent_div: np.ndarray,
    t_plus_1: float,
    w: int,
    nt_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the sliding window of observed pi and expected pi.

    For each index, symmetrically expand a window of neighbouring indices
    until the cumulative silent-site count reaches ``w`` (or the sequence
    ends). Within that window,

        obs_pi   = sum(silent_pi) / sum(silent_sites)
        exp_pi   = (sum(silent_div) / sum(silent_sites)) / (T + 1)

    Windows are assigned to the nucleotide position supplied in
    ``nt_positions[i]``; when omitted, the function assumes per-codon input
    (each index is one codon, 3 nt apart) and uses ``codon_nt_position``.

    Args:
        silent_sites: Per-index silent-site count (length N).
        silent_pi:   Per-index ingroup mean silent pairwise differences.
        silent_div:  Per-index ingroup-vs-outgroup mean silent divergences.
        t_plus_1:    Global divergence-time scaler (see ``sliding_hka.hka``).
        w:           Target window width in silent sites.
        nt_positions: Optional per-index nucleotide positions for plotting
            (length N). Defaults to codon-midpoint mapping (3*i + 1).

    Returns:
        Dict with numpy arrays ``nt_position``, ``obs_pi``, ``exp_pi``, and
        ``sites_in_window`` (length N each).
    """
    n = len(silent_sites)
    if len(silent_pi) != n or len(silent_div) != n:
        raise ValueError("silent_sites, silent_pi, silent_div must be same length")
    if w <= 0:
        raise ValueError("w must be positive")
    if t_plus_1 <= 0:
        raise ValueError("t_plus_1 must be positive")

    if nt_positions is None:
        nt_pos = np.array([codon_nt_position(i) for i in range(n)], dtype=int)
    else:
        if len(nt_positions) != n:
            raise ValueError(
                f"nt_positions length {len(nt_positions)} != array length {n}"
            )
        nt_pos = np.asarray(nt_positions, dtype=int)
    obs = np.full(n, np.nan)
    exp = np.full(n, np.nan)
    sites_in_window = np.zeros(n)

    for c in range(n):
        # Centers that contribute no silent sites themselves sit inside an
        # alignment dead zone; any window anchored there is dominated by
        # distant active positions and produces flat plateaus across the
        # whole dead zone. Refuse those centers — they have no local signal.
        if silent_sites[c] <= 0:
            continue

        left = right = c
        total_sites = float(silent_sites[c])
        blocked = False
        while total_sites < w:
            can_left = left > 0
            can_right = right < n - 1
            # If either side is blocked we can no longer expand symmetrically
            # around the center. Borrowing from the open side would produce
            # an asymmetric window whose "center" is not the codon c we are
            # assigning the value to — and repeated borrows across adjacent
            # edge centers would all converge to the same window, flattening
            # the plot. Mark the center invalid instead.
            if not can_left or not can_right:
                blocked = True
                break
            left_span = c - left
            right_span = right - c
            if left_span <= right_span:
                left -= 1
                total_sites += float(silent_sites[left])
            else:
                right += 1
                total_sites += float(silent_sites[right])

        if blocked or total_sites < w:
            continue

        sum_sites = float(silent_sites[left : right + 1].sum())
        sites_in_window[c] = sum_sites
        if sum_sites > 0:
            sum_pi = float(silent_pi[left : right + 1].sum())
            sum_div = float(silent_div[left : right + 1].sum())
            obs[c] = sum_pi / sum_sites
            exp[c] = (sum_div / sum_sites) / t_plus_1

    return {
        "nt_position": nt_pos,
        "obs_pi": obs,
        "exp_pi": exp,
        "sites_in_window": sites_in_window,
    }
