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
) -> dict[str, np.ndarray]:
    """Compute the sliding window of observed pi and expected pi.

    For each codon index, symmetrically expand a window of neighbouring
    codons until the cumulative silent-site count reaches ``w`` (or the
    sequence ends). Within that window,

        obs_pi   = sum(silent_pi) / sum(silent_sites)
        exp_pi   = (sum(silent_div) / sum(silent_sites)) / (T + 1)

    Windows are assigned to the nucleotide midpoint of the center codon.

    Args:
        silent_sites: Per-codon Nei-Gojobori synonymous-site count (length N).
        silent_pi:   Per-codon mean ingroup pairwise silent differences.
        silent_div:  Per-codon mean ingroup-vs-outgroup silent divergences.
        t_plus_1:    Global divergence-time scaler (see ``sliding_hka.hka``).
        w:           Target window width in silent sites.

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

    nt_pos = np.array([codon_nt_position(i) for i in range(n)], dtype=int)
    obs = np.full(n, np.nan)
    exp = np.full(n, np.nan)
    sites_in_window = np.zeros(n)

    for c in range(n):
        left = right = c
        total_sites = float(silent_sites[c])
        while total_sites < w:
            can_left = left > 0
            can_right = right < n - 1
            if not can_left and not can_right:
                break
            # Expand the shorter side first to stay symmetric around c.
            left_span = c - left
            right_span = right - c
            extend_left = can_left and (not can_right or left_span <= right_span)
            if extend_left:
                left -= 1
                total_sites += float(silent_sites[left])
            else:
                right += 1
                total_sites += float(silent_sites[right])

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
