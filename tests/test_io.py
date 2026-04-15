"""Tests for sliding_hka.io."""

from __future__ import annotations

from pathlib import Path

import pytest

from sliding_hka.io import load_msa

FIXTURES = Path(__file__).parent / "data"


def test_load_msa_returns_ingroup_outgroup_sequencesets():
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    assert len(ingroup) == 4
    assert len(outgroup) == 1


def test_load_msa_default_prefixes_route_sequences():
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    assert all(s.name.startswith("Bgland_") for s in ingroup.sequences)
    assert all(s.name.startswith("Bcrena_") for s in outgroup.sequences)


def test_load_msa_custom_prefixes(tmp_path: Path):
    fa = tmp_path / "custom.fa"
    fa.write_text(">Out_a\nATGATG\n>In_1\nATGATG\n>In_2\nATGATG\n")
    ingroup, outgroup = load_msa(fa, ingroup_prefix="In_", outgroup_prefix="Out_")
    assert len(ingroup) == 2
    assert len(outgroup) == 1


def test_load_msa_raises_on_no_outgroup(tmp_path: Path):
    fa = tmp_path / "no_out.fa"
    fa.write_text(">Bgland_1\nATGATG\n>Bgland_2\nATGATG\n")
    with pytest.raises(ValueError, match="outgroup"):
        load_msa(fa)


def test_load_msa_raises_on_multiple_outgroups_by_default(tmp_path: Path):
    fa = tmp_path / "two_out.fa"
    fa.write_text(
        ">Bcrena_a\nATGATG\n"
        ">Bcrena_b\nATGATG\n"
        ">Bgland_1\nATGATG\n"
    )
    with pytest.raises(ValueError, match="multiple outgroup"):
        load_msa(fa)


def test_load_msa_allows_multi_outgroup_when_flagged(tmp_path: Path):
    fa = tmp_path / "two_out.fa"
    fa.write_text(
        ">Bcrena_a\nATGATG\n"
        ">Bcrena_b\nATGATG\n"
        ">Bgland_1\nATGATG\n"
    )
    ingroup, outgroup = load_msa(fa, allow_multi_outgroup=True)
    assert len(outgroup) == 2


def test_load_msa_raises_on_length_mismatch(tmp_path: Path):
    fa = tmp_path / "mismatched.fa"
    fa.write_text(">Bcrena_a\nATGATG\n>Bgland_1\nATG\n")
    with pytest.raises(ValueError, match="length"):
        load_msa(fa)


def test_load_msa_raises_on_length_not_multiple_of_three(tmp_path: Path):
    fa = tmp_path / "notriplet.fa"
    fa.write_text(">Bcrena_a\nATGA\n>Bgland_1\nATGA\n")
    with pytest.raises(ValueError, match="multiple of three"):
        load_msa(fa)
