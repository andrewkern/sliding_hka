"""Tests for sliding_hka.io."""

from __future__ import annotations

from pathlib import Path

import pytest

from sliding_hka.io import load_msa

FIXTURES = Path(__file__).parent / "data"


def test_default_treats_first_sequence_as_outgroup(tmp_path: Path):
    fa = tmp_path / "first_is_out.fa"
    fa.write_text(
        ">outgroup_seq\nATGATG\n"
        ">sample_a\nATGATG\n"
        ">sample_b\nATGATG\n"
    )
    ingroup, outgroup = load_msa(fa)
    assert len(outgroup) == 1
    assert outgroup.sequences[0].name == "outgroup_seq"
    assert len(ingroup) == 2


def test_default_works_on_synthetic_min_fixture():
    # Bcrena is listed first in the fixture -> 1 outgroup + 4 ingroup.
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    assert len(outgroup) == 1
    assert len(ingroup) == 4
    assert outgroup.sequences[0].name.startswith("Bcrena_")
    assert all(s.name.startswith("Bgland_") for s in ingroup.sequences)


def test_outgroup_match_substring_routes_correctly(tmp_path: Path):
    fa = tmp_path / "mixed.fa"
    fa.write_text(
        ">sample_a\nATGATG\n"
        ">outgroup_x\nATGATG\n"
        ">sample_b\nATGATG\n"
    )
    ingroup, outgroup = load_msa(fa, outgroup_match="outgroup")
    assert len(outgroup) == 1
    assert outgroup.sequences[0].name == "outgroup_x"
    assert len(ingroup) == 2


def test_ingroup_match_alone_defaults_outgroup_to_rest(tmp_path: Path):
    fa = tmp_path / "mixed.fa"
    fa.write_text(
        ">sample_a\nATGATG\n"
        ">outgroup_x\nATGATG\n"
        ">sample_b\nATGATG\n"
    )
    ingroup, outgroup = load_msa(fa, ingroup_match="sample")
    assert len(ingroup) == 2
    assert len(outgroup) == 1
    assert outgroup.sequences[0].name == "outgroup_x"


def test_both_matches_restrict_both_groups(tmp_path: Path):
    fa = tmp_path / "mixed.fa"
    fa.write_text(
        ">sample_a\nATGATG\n"
        ">outgroup_x\nATGATG\n"
        ">junk_z\nATGATG\n"
    )
    ingroup, outgroup = load_msa(fa, ingroup_match="sample", outgroup_match="outgroup")
    assert len(ingroup) == 1
    assert len(outgroup) == 1


def test_raises_when_outgroup_match_matches_nothing(tmp_path: Path):
    fa = tmp_path / "mixed.fa"
    fa.write_text(">a\nATGATG\n>b\nATGATG\n")
    with pytest.raises(ValueError, match="outgroup"):
        load_msa(fa, outgroup_match="nope")


def test_raises_on_multiple_outgroups_by_default(tmp_path: Path):
    fa = tmp_path / "two_out.fa"
    fa.write_text(
        ">outgroup_a\nATGATG\n"
        ">outgroup_b\nATGATG\n"
        ">sample_1\nATGATG\n"
    )
    with pytest.raises(ValueError, match="outgroup"):
        load_msa(fa, outgroup_match="outgroup")


def test_allows_multi_outgroup_when_flagged(tmp_path: Path):
    fa = tmp_path / "two_out.fa"
    fa.write_text(
        ">outgroup_a\nATGATG\n"
        ">outgroup_b\nATGATG\n"
        ">sample_1\nATGATG\n"
    )
    ingroup, outgroup = load_msa(
        fa, outgroup_match="outgroup", allow_multi_outgroup=True
    )
    assert len(outgroup) == 2
    assert len(ingroup) == 1


def test_raises_on_length_mismatch(tmp_path: Path):
    fa = tmp_path / "mismatched.fa"
    fa.write_text(">a\nATGATG\n>b\nATG\n")
    with pytest.raises(ValueError, match="length"):
        load_msa(fa)


def test_load_msa_accepts_non_triplet_length(tmp_path: Path):
    # io.load_msa no longer enforces multiple-of-three; that constraint is
    # applied only when callers actually require codon-framed analysis
    # (e.g. counts.per_codon_arrays).
    fa = tmp_path / "notriplet.fa"
    fa.write_text(">a\nATGA\n>b\nATGA\n")
    ingroup, outgroup = load_msa(fa)
    assert len(ingroup) == 1
    assert len(outgroup) == 1
