"""Tests for sliding_hka.hka."""

from __future__ import annotations

import pytest

from sliding_hka.hka import LocusTotals, estimate_t_plus_1


def test_t_plus_1_single_locus_matches_paper_example():
    # Paper's Adh region estimate was T+1 = 6.3.
    # With pi_total = 0.01 and div_total = 0.063, T+1 = 6.3.
    totals = [LocusTotals(silent_pi_sum=0.01, silent_div_sum=0.063, silent_sites_sum=1.0)]
    assert estimate_t_plus_1(totals) == pytest.approx(6.3)


def test_t_plus_1_is_ratio_of_per_site_rates():
    # pi_total / sites = 0.5, div_total / sites = 1.0 -> T+1 = 2.0
    totals = [LocusTotals(silent_pi_sum=5.0, silent_div_sum=10.0, silent_sites_sum=10.0)]
    assert estimate_t_plus_1(totals) == pytest.approx(2.0)


def test_t_plus_1_joint_pools_across_loci():
    # Three loci; pooled totals: pi=3, div=6, sites=6 -> per-site pi=0.5, div=1.0, T+1=2
    totals = [
        LocusTotals(silent_pi_sum=1.0, silent_div_sum=2.0, silent_sites_sum=2.0),
        LocusTotals(silent_pi_sum=1.0, silent_div_sum=2.0, silent_sites_sum=2.0),
        LocusTotals(silent_pi_sum=1.0, silent_div_sum=2.0, silent_sites_sum=2.0),
    ]
    assert estimate_t_plus_1(totals) == pytest.approx(2.0)


def test_t_plus_1_raises_on_zero_polymorphism():
    totals = [LocusTotals(silent_pi_sum=0.0, silent_div_sum=1.0, silent_sites_sum=10.0)]
    with pytest.raises(ValueError, match="polymorphism"):
        estimate_t_plus_1(totals)


def test_t_plus_1_raises_on_zero_sites():
    totals = [LocusTotals(silent_pi_sum=0.0, silent_div_sum=0.0, silent_sites_sum=0.0)]
    with pytest.raises(ValueError, match="silent sites"):
        estimate_t_plus_1(totals)


def test_locus_totals_from_arrays_sums_correctly():
    # LocusTotals convenience constructor from per-codon arrays.
    import numpy as np

    sites = np.array([1.0, 0.0, 1.0])
    pi = np.array([0.2, 0.0, 0.4])
    div = np.array([0.5, 0.0, 1.0])
    t = LocusTotals.from_arrays(sites, pi, div)
    assert t.silent_pi_sum == pytest.approx(0.6)
    assert t.silent_div_sum == pytest.approx(1.5)
    assert t.silent_sites_sum == pytest.approx(2.0)
