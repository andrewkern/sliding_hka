"""Tests for sliding_hka.window."""

from __future__ import annotations

import numpy as np
import pytest

from sliding_hka.window import codon_nt_position, sliding_window


def test_codon_nt_position_uses_codon_midpoint():
    # codon i starts at base 3*i; midpoint base is 3*i + 1 (0-indexed)
    assert codon_nt_position(0) == 1
    assert codon_nt_position(5) == 16


def test_window_returns_one_entry_per_codon():
    sites = np.ones(5)
    pi = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    div = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    out = sliding_window(sites, pi, div, t_plus_1=2.0, w=3)
    assert len(out["nt_position"]) == 5


def test_window_nt_positions_are_codon_midpoints():
    sites = np.ones(4)
    pi = np.zeros(4)
    div = np.zeros(4)
    out = sliding_window(sites, pi, div, t_plus_1=1.0, w=1)
    assert list(out["nt_position"]) == [1, 4, 7, 10]


def test_window_holds_at_least_w_silent_sites_in_middle():
    # Each codon contributes 1 silent site; w=3 should cover 3 codons.
    sites = np.ones(7)
    pi = np.zeros(7)
    div = np.zeros(7)
    out = sliding_window(sites, pi, div, t_plus_1=1.0, w=3)
    # Middle codon (index 3) has full window of 3 sites
    assert out["sites_in_window"][3] >= 3


def test_window_observed_pi_hand_calc():
    # 5 codons each with 1 silent site; pi = [0,1,2,1,0]; w=3.
    # Center codon 2: window covers codons 1,2,3. obs_pi = (1+2+1)/3 = 4/3
    sites = np.ones(5)
    pi = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    div = np.zeros(5)
    out = sliding_window(sites, pi, div, t_plus_1=1.0, w=3)
    assert out["obs_pi"][2] == pytest.approx(4 / 3)


def test_window_expected_pi_hand_calc():
    # Same window. div=[1,1,1,1,1], T+1=2.
    # exp_pi center = (1+1+1)/3 / 2 = 0.5
    sites = np.ones(5)
    pi = np.zeros(5)
    div = np.ones(5)
    out = sliding_window(sites, pi, div, t_plus_1=2.0, w=3)
    assert out["exp_pi"][2] == pytest.approx(0.5)


def test_window_at_left_edge_truncates():
    sites = np.ones(5)
    pi = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    div = np.zeros(5)
    out = sliding_window(sites, pi, div, t_plus_1=1.0, w=3)
    # Codon 0 can extend right but not left; window covers <= 3 sites
    assert out["sites_in_window"][0] <= 3
    # Observed pi should still be finite
    assert np.isfinite(out["obs_pi"][0])


def test_window_handles_zero_sites_in_some_codons():
    # Codons with no silent sites (e.g. dirty outgroup) contribute 0 to width
    sites = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    pi = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    div = np.zeros(5)
    out = sliding_window(sites, pi, div, t_plus_1=1.0, w=2)
    # Center codon 2: needs >=2 silent sites; codons 0,2,4 contribute 1 each
    assert out["sites_in_window"][2] >= 2
    assert np.isfinite(out["obs_pi"][2])


def test_window_global_pi_and_div_returned():
    sites = np.array([1.0, 1.0, 1.0])
    pi = np.array([0.5, 0.5, 0.5])
    div = np.array([1.0, 1.0, 1.0])
    out = sliding_window(sites, pi, div, t_plus_1=2.0, w=1)
    # Each window of 1 site at center yields obs = pi/sites = 0.5
    assert out["obs_pi"][1] == pytest.approx(0.5)
    # exp = (1/1) / 2 = 0.5
    assert out["exp_pi"][1] == pytest.approx(0.5)
