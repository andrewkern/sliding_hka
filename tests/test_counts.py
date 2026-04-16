"""Tests for sliding_hka.counts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mkado.core.codons import DEFAULT_CODE

from sliding_hka.counts import (
    count_segregating_silent,
    per_codon_arrays,
    segregating_silent_codon,
    silent_divergence_codon,
    silent_pairwise_diff_codon,
    silent_sites_codon,
)
from sliding_hka.io import load_msa

FIXTURES = Path(__file__).parent / "data"


# --- silent_sites_codon ---


def test_silent_sites_met_is_zero():
    assert silent_sites_codon("ATG") == 0.0


def test_silent_sites_phe_matches_ng():
    assert silent_sites_codon("TTC") == pytest.approx(1 / 3)


def test_silent_sites_ala_is_one():
    assert silent_sites_codon("GCG") == pytest.approx(1.0)


def test_silent_sites_gapped_codon_returns_zero():
    assert silent_sites_codon("A-G") == 0.0
    assert silent_sites_codon("NNN") == 0.0


# --- silent_pairwise_diff_codon ---


def test_pairwise_diff_two_seqs_one_syn_change():
    # TTT and TTC differ by a single synonymous change.
    mean_s, n_pairs = silent_pairwise_diff_codon(["TTT", "TTC"])
    assert mean_s == pytest.approx(1.0)
    assert n_pairs == 1


def test_pairwise_diff_two_seqs_replacement_only():
    # TTT (Phe) and TTA (Leu) differ by a replacement; no silent diffs.
    mean_s, n_pairs = silent_pairwise_diff_codon(["TTT", "TTA"])
    assert mean_s == pytest.approx(0.0)
    assert n_pairs == 1


def test_pairwise_diff_multi_seqs_mean():
    # Ingroup has 2 TTT + 2 TTC. 6 pairs, 4 differ by 1 syn change each.
    mean_s, n_pairs = silent_pairwise_diff_codon(["TTT", "TTT", "TTC", "TTC"])
    assert mean_s == pytest.approx(4 / 6)
    assert n_pairs == 6


def test_pairwise_diff_excludes_gapped_codons():
    # Only clean codons participate; dirty ones are dropped before pairing.
    mean_s, n_pairs = silent_pairwise_diff_codon(["TTT", "TTC", "T-T", "NNN"])
    assert n_pairs == 1
    assert mean_s == pytest.approx(1.0)


def test_pairwise_diff_all_invariant_is_zero():
    mean_s, n_pairs = silent_pairwise_diff_codon(["ATG", "ATG", "ATG"])
    assert mean_s == pytest.approx(0.0)
    assert n_pairs == 3


def test_pairwise_diff_no_clean_pairs_returns_nan():
    # One clean codon → no pairs possible; function should signal with NaN.
    mean_s, n_pairs = silent_pairwise_diff_codon(["---", "NNN", "ATG"])
    assert n_pairs == 0
    assert math.isnan(mean_s)


# --- silent_divergence_codon ---


def test_divergence_one_ingroup_one_syn_diff():
    mean_s, n_obs = silent_divergence_codon(["TTT"], "TTC")
    assert mean_s == pytest.approx(1.0)
    assert n_obs == 1


def test_divergence_multi_ingroup_mean():
    # outgroup TTC, ingroup [TTT, TTT, TTC, TTC] → 2 of 4 differ by 1 syn
    mean_s, n_obs = silent_divergence_codon(["TTT", "TTT", "TTC", "TTC"], "TTC")
    assert mean_s == pytest.approx(0.5)
    assert n_obs == 4


def test_divergence_skips_gapped_ingroup():
    mean_s, n_obs = silent_divergence_codon(["TTT", "T-T"], "TTC")
    assert n_obs == 1
    assert mean_s == pytest.approx(1.0)


def test_divergence_dirty_outgroup_returns_nan():
    mean_s, n_obs = silent_divergence_codon(["TTT", "TTC"], "NNN")
    assert n_obs == 0
    assert math.isnan(mean_s)


# --- per_codon_arrays: integration against synthetic_min.fa ---


def test_per_codon_arrays_against_synthetic_min():
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    sites, pi, div = per_codon_arrays(ingroup, outgroup)

    assert len(sites) == len(pi) == len(div) == 4

    # Codon 0: ATG invariant
    assert sites[0] == pytest.approx(0.0)
    assert pi[0] == pytest.approx(0.0)
    assert div[0] == pytest.approx(0.0)

    # Codon 1: TTC outgroup, ingroup [TTT,TTT,TTC,TTC]
    assert sites[1] == pytest.approx(DEFAULT_CODE.count_synonymous_sites("TTC"))
    assert pi[1] == pytest.approx(4 / 6)
    assert div[1] == pytest.approx(0.5)

    # Codon 2: GCG invariant
    assert sites[2] == pytest.approx(1.0)
    assert pi[2] == pytest.approx(0.0)
    assert div[2] == pytest.approx(0.0)

    # Codon 3: CGC outgroup, ingroup [CGC,CGC,CGT,CGC]
    assert sites[3] == pytest.approx(1.0)
    assert pi[3] == pytest.approx(3 / 6)
    assert div[3] == pytest.approx(1 / 4)


def test_per_codon_arrays_handles_dirty_codons():
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_edges.fa")
    sites, pi, div = per_codon_arrays(ingroup, outgroup)
    assert len(sites) == 6

    # Codon 2: outgroup NNN -> everything zeroed
    assert sites[2] == pytest.approx(0.0)
    assert pi[2] == pytest.approx(0.0)
    assert div[2] == pytest.approx(0.0)

    # Codon 3: outgroup CGC, ingroup [---, CGC, CGT, CGC] -> 3 clean
    # Clean pairs: (CGC,CGT)=1S, (CGC,CGC)=0, (CGT,CGC)=1S -> mean 2/3
    assert sites[3] == pytest.approx(1.0)
    assert pi[3] == pytest.approx(2 / 3)
    assert div[3] == pytest.approx(1 / 3)


# --- segregating_silent_codon ---


def test_segregating_silent_codon_two_syn_alleles():
    # TTT and TTC differ by one synonymous change -> 1 segregating silent site
    assert segregating_silent_codon(["TTT", "TTC"]) == 1


def test_segregating_silent_codon_replacement_only():
    # TTT and TTA differ by one replacement change -> 0 segregating silent sites
    assert segregating_silent_codon(["TTT", "TTA"]) == 0


def test_segregating_silent_codon_invariant():
    assert segregating_silent_codon(["ATG", "ATG", "ATG"]) == 0


def test_segregating_silent_codon_skips_dirty():
    # Only one clean codon -> 0 (can't segregate with one sequence)
    assert segregating_silent_codon(["TTT", "---", "NNN"]) == 0


# --- count_segregating_silent (integration) ---


def test_count_segregating_silent_synthetic_min():
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    s = count_segregating_silent(ingroup, outgroup)
    # synthetic_min.fa codons:
    #   codon 0: ATG invariant -> 0
    #   codon 1: ingroup [TTT,TTT,TTC,TTC] -> 1 syn segregating site
    #   codon 2: GCG invariant -> 0
    #   codon 3: ingroup [CGC,CGC,CGT,CGC] -> 1 syn segregating site
    assert s == 2
