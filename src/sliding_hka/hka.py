"""HKA divergence-time scaler estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LocusTotals:
    """Pooled silent-site, silent-pi, and silent-divergence sums for a locus."""

    silent_pi_sum: float
    silent_div_sum: float
    silent_sites_sum: float

    @classmethod
    def from_arrays(
        cls,
        silent_sites: np.ndarray,
        silent_pi: np.ndarray,
        silent_div: np.ndarray,
    ) -> LocusTotals:
        return cls(
            silent_pi_sum=float(silent_pi.sum()),
            silent_div_sum=float(silent_div.sum()),
            silent_sites_sum=float(silent_sites.sum()),
        )


def estimate_t_plus_1(totals: Sequence[LocusTotals]) -> float:
    """Estimate the divergence-time scaler T + 1 = D / pi.

    With one locus, this is simply the ratio of per-site silent divergence
    to per-site silent polymorphism. With multiple loci, sums are pooled
    across loci before taking the ratio (jointly over all regions, as in
    the HKA framework when the loci share a single ancestral Ne and
    divergence time).

    Args:
        totals: One ``LocusTotals`` per locus.

    Returns:
        The scalar T + 1.

    Raises:
        ValueError: if total silent sites is zero, or if total silent
        polymorphism is zero (which would divide by zero).
    """
    sites = sum(t.silent_sites_sum for t in totals)
    pi_sum = sum(t.silent_pi_sum for t in totals)
    div_sum = sum(t.silent_div_sum for t in totals)

    if sites <= 0:
        raise ValueError("total silent sites is zero; cannot estimate T+1")
    if pi_sum <= 0:
        raise ValueError(
            "total silent polymorphism is zero; cannot estimate T+1"
        )
    pi_rate = pi_sum / sites
    div_rate = div_sum / sites
    return div_rate / pi_rate
