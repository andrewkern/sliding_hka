"""Tests for sliding_hka.annotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sliding_hka.annotation import LocusAnnotation


def test_empty_annotation():
    ann = LocusAnnotation.empty(10)
    assert ann.n_positions == 10
    assert all(f == "" for f in ann.feature)
    assert all(c == -1 for c in ann.codon_index)
    assert all(p == -1 for p in ann.codon_pos)
    assert not ann.is_cds().any()


def test_construct_from_arrays():
    feat = np.array(["intergenic", "intergenic", "CDS", "CDS", "CDS", "intron", "CDS", "CDS", "CDS"], dtype="U16")
    cidx = np.array([-1, -1, 0, 0, 0, -1, 1, 1, 1], dtype=np.int64)
    cpos = np.array([-1, -1, 0, 1, 2, -1, 0, 1, 2], dtype=np.int8)
    ann = LocusAnnotation(n_positions=9, feature=feat, codon_index=cidx, codon_pos=cpos)
    assert ann.is_cds().sum() == 6
    assert not ann.is_cds()[0]
    assert ann.is_cds()[2]


def test_codon_groups_returns_position_triplets():
    feat = np.array(["CDS"] * 6 + ["intron"] + ["CDS"] * 3, dtype="U16")
    cidx = np.array([0, 0, 0, 1, 1, 1, -1, 2, 2, 2], dtype=np.int64)
    cpos = np.array([0, 1, 2, 0, 1, 2, -1, 0, 1, 2], dtype=np.int8)
    ann = LocusAnnotation(n_positions=10, feature=feat, codon_index=cidx, codon_pos=cpos)
    groups = ann.codon_groups()
    assert set(groups.keys()) == {0, 1, 2}
    assert groups[0] == (0, 1, 2)
    assert groups[1] == (3, 4, 5)
    assert groups[2] == (7, 8, 9)


def test_codon_groups_handles_intron_split_codon():
    # codon 0 spans positions 0,1, then jumps over an intron at 2 to position 3
    feat = np.array(["CDS", "CDS", "intron", "CDS", "CDS", "CDS", "CDS"], dtype="U16")
    cidx = np.array([0, 0, -1, 0, 1, 1, 1], dtype=np.int64)
    cpos = np.array([0, 1, -1, 2, 0, 1, 2], dtype=np.int8)
    ann = LocusAnnotation(n_positions=7, feature=feat, codon_index=cidx, codon_pos=cpos)
    groups = ann.codon_groups()
    assert groups[0] == (0, 1, 3)
    assert groups[1] == (4, 5, 6)


def test_from_tsv_parses_per_position_annotation(tmp_path: Path):
    tsv = tmp_path / "ann.tsv"
    tsv.write_text(
        "pos_1based\tfeature\tcodon_index\tcodon_pos\n"
        "1\tintergenic\t-1\t-1\n"
        "2\tCDS\t0\t0\n"
        "3\tCDS\t0\t1\n"
        "4\tCDS\t0\t2\n"
        "5\tintron\t-1\t-1\n"
    )
    ann = LocusAnnotation.from_tsv(tsv, alignment_length=5)
    assert ann.n_positions == 5
    assert ann.feature[1] == "CDS"
    assert ann.codon_index[3] == 0
    assert ann.codon_pos[3] == 2


def test_from_tsv_raises_on_length_mismatch(tmp_path: Path):
    tsv = tmp_path / "ann.tsv"
    tsv.write_text("pos_1based\tfeature\tcodon_index\tcodon_pos\n1\tintergenic\t-1\t-1\n")
    with pytest.raises(ValueError, match="length"):
        LocusAnnotation.from_tsv(tsv, alignment_length=10)
