"""Tests for sliding_hka.counts.per_position_arrays (annotated mode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mkado.core.sequences import Sequence, SequenceSet

from sliding_hka.annotation import LocusAnnotation
from sliding_hka.counts import (
    count_segregating_silent_annotated,
    per_codon_arrays,
    per_position_arrays,
)

FIXTURES = Path(__file__).parent / "data"


def _seq_set(named: dict[str, str]) -> SequenceSet:
    return SequenceSet(sequences=[Sequence(name=n, sequence=s) for n, s in named.items()])


def _annotate(ann: LocusAnnotation, pos: int, feature: str,
              codon_index: int = -1, codon_pos: int = -1) -> None:
    ann.feature[pos] = feature
    ann.codon_index[pos] = codon_index
    ann.codon_pos[pos] = codon_pos


# --- CDS-only annotation: totals should match per_codon_arrays ---


def test_per_position_cds_only_totals_match_per_codon_totals():
    """Plus-strand, every position CDS, contiguous codons."""
    from sliding_hka.io import load_msa
    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")

    # Build a CDS-only annotation: 4 codons × 3 nt = 12 positions
    ann = LocusAnnotation.empty(12, strand="+")
    for pos in range(12):
        _annotate(ann, pos, "CDS", codon_index=pos // 3, codon_pos=pos % 3)

    sites_c, pi_c, div_c = per_codon_arrays(ingroup, outgroup)
    sites_p, pi_p, div_p = per_position_arrays(ingroup, outgroup, ann)

    # Totals must match exactly (same information, redistributed)
    assert sites_p.sum() == pytest.approx(sites_c.sum())
    assert pi_p.sum() == pytest.approx(pi_c.sum())
    assert div_p.sum() == pytest.approx(div_c.sum())


def test_per_position_cds_codon_split_across_intron():
    """Codon 0 spans positions 0, 1, 3 (with an intron base at 2).

    Reading positions {0, 1, 3} of each sequence:
        outgroup    -> 'TTC'  (Phe)
        Bgland_1_1  -> 'TTT'  (Phe; synonymous)
        Bgland_1_2  -> 'TTC'  (Phe; identical to out)
    """
    ing = _seq_set({
        "Bgland_1_1": "TTAT",
        "Bgland_1_2": "TTAC",
    })
    out = _seq_set({
        "Bcrena_01_1": "TTAC",
    })
    ann = LocusAnnotation.empty(4, strand="+")
    _annotate(ann, 0, "CDS", codon_index=0, codon_pos=0)
    _annotate(ann, 1, "CDS", codon_index=0, codon_pos=1)
    _annotate(ann, 2, "intron")
    _annotate(ann, 3, "CDS", codon_index=0, codon_pos=2)

    sites_p, pi_p, div_p = per_position_arrays(ing, out, ann)

    # Codon 0 (TTC outgroup, ingroup TTT/TTC): silent_sites=NG('TTC')=1/3,
    # pi (one pair, syn diff)=1.0, div (1 of 2 pairs differ by syn)=0.5.
    total_cds_sites = sites_p[[0, 1, 3]].sum()
    total_cds_pi = pi_p[[0, 1, 3]].sum()
    total_cds_div = div_p[[0, 1, 3]].sum()
    assert total_cds_sites == pytest.approx(1 / 3)
    assert total_cds_pi == pytest.approx(1.0)
    assert total_cds_div == pytest.approx(0.5)

    # Intron position should be treated as per-site silent.
    assert sites_p[2] == pytest.approx(1.0)


# --- Non-CDS annotation: per-site nt diversity ---


def test_per_position_non_cds_all_silent():
    """Every position marked non-CDS → every alignable column counts as 1 silent site."""
    ing = _seq_set({
        "Bgland_1_1": "ACGT",
        "Bgland_1_2": "ACGA",
        "Bgland_2_1": "ACGT",
        "Bgland_2_2": "ACGA",
    })
    out = _seq_set({
        "Bcrena_01_1": "ACGT",
    })
    ann = LocusAnnotation.empty(4, strand="+")
    for p in range(4):
        _annotate(ann, p, "intergenic")

    sites, pi, div = per_position_arrays(ing, out, ann)
    assert list(sites) == [1.0, 1.0, 1.0, 1.0]

    # Position 3: ingroup [T, A, T, A] → 4 diffs / 6 pairs = 2/3.
    assert pi[3] == pytest.approx(4 / 6)
    # Positions 0-2: invariant ingroup -> pi = 0
    assert pi[0] == 0.0
    # Position 3 div: ingroup [T, A, T, A] vs outgroup [T]: 2 diffs / 4 obs = 0.5
    assert div[3] == pytest.approx(0.5)


def test_per_position_non_cds_skips_when_outgroup_gapped():
    ing = _seq_set({
        "Bgland_1_1": "ACGT",
        "Bgland_2_1": "ACGA",
    })
    out = _seq_set({
        "Bcrena_01_1": "AC-T",
    })
    ann = LocusAnnotation.empty(4, strand="+")
    for p in range(4):
        _annotate(ann, p, "intergenic")
    sites, pi, div = per_position_arrays(ing, out, ann)
    assert sites[2] == 0.0
    assert pi[2] == 0.0
    assert div[2] == 0.0


# --- Mixed annotation ---


def test_per_position_mixed_cds_and_intergenic():
    """1 CDS codon + 2 intergenic positions."""
    ing = _seq_set({
        "Bgland_1_1": "TTTAG",
        "Bgland_1_2": "TTCAG",
    })
    out = _seq_set({
        "Bcrena_01_1": "TTCAT",
    })
    ann = LocusAnnotation.empty(5, strand="+")
    for i in range(3):
        _annotate(ann, i, "CDS", codon_index=0, codon_pos=i)
    _annotate(ann, 3, "intergenic")
    _annotate(ann, 4, "intergenic")

    sites, pi, div = per_position_arrays(ing, out, ann)

    # CDS codon: out=TTC, in=[TTT,TTC]. silent_sites=NG('TTC')=1/3, pi=1.0, div=0.5
    assert sites[:3].sum() == pytest.approx(1 / 3)
    assert pi[:3].sum() == pytest.approx(1.0)
    assert div[:3].sum() == pytest.approx(0.5)

    # Intergenic col 3: invariant AA, outgroup A → pi=0, div=0, sites=1
    assert sites[3] == 1.0
    assert pi[3] == 0.0
    assert div[3] == 0.0
    # Intergenic col 4: ingroup all G, outgroup T → div=1
    assert sites[4] == 1.0
    assert pi[4] == 0.0
    assert div[4] == 1.0


# --- Minus-strand handling ---


def test_per_position_cds_minus_strand_revcomps_codon():
    """Minus-strand gene: alignment bases are +strand; codon is their revcomp."""
    # On the + strand (alignment) we have 'GAA' but the CDS on - strand is 'TTC'.
    # Ingroup: + strand 'GAA'/'AAA' -> - strand codons 'TTC'/'TTT'. So it's an
    # Adh-like F↔F synonymous (TTC/TTT) polymorphism.
    ing = _seq_set({
        "Bgland_1_1": "GAA",
        "Bgland_1_2": "AAA",
    })
    out = _seq_set({
        "Bcrena_01_1": "GAA",  # - strand 'TTC'
    })
    ann = LocusAnnotation.empty(3, strand="-")
    for i in range(3):
        _annotate(ann, i, "CDS", codon_index=0, codon_pos=i)

    sites, pi, div = per_position_arrays(ing, out, ann)
    # Codon on - strand: TTC. ingroup [TTT,TTC], outgroup TTC -> 1 pi, 0.5 div
    assert sites.sum() == pytest.approx(1 / 3)
    assert pi.sum() == pytest.approx(1.0)
    assert div.sum() == pytest.approx(0.5)


# --- count_segregating_silent_annotated ---


def test_seg_annotated_cds_only():
    """CDS-only annotation: should match count_segregating_silent."""
    from sliding_hka.counts import count_segregating_silent
    from sliding_hka.io import load_msa

    ingroup, outgroup = load_msa(FIXTURES / "synthetic_min.fa")
    ann = LocusAnnotation.empty(12, strand="+")
    for pos in range(12):
        _annotate(ann, pos, "CDS", codon_index=pos // 3, codon_pos=pos % 3)

    seg_codon = count_segregating_silent(ingroup, outgroup)
    seg_ann = count_segregating_silent_annotated(ingroup, outgroup, ann)
    assert seg_ann == seg_codon


def test_seg_annotated_non_cds_counts_nt_segregation():
    """Non-CDS positions segregating at nucleotide level should be counted."""
    ing = _seq_set({
        "Bgland_1_1": "TTTAG",
        "Bgland_1_2": "TTCAG",
    })
    out = _seq_set({
        "Bcrena_01_1": "TTCAT",
    })
    ann = LocusAnnotation.empty(5, strand="+")
    for i in range(3):
        _annotate(ann, i, "CDS", codon_index=0, codon_pos=i)
    _annotate(ann, 3, "intergenic")
    _annotate(ann, 4, "intergenic")

    seg = count_segregating_silent_annotated(ing, out, ann)
    # CDS codon TTT/TTC: 1 syn segregating site.
    # Intergenic pos 3: A/A invariant -> 0.
    # Intergenic pos 4: G/G invariant -> 0.
    assert seg == 1


def test_seg_annotated_mixed_with_noncds_polymorphism():
    """Non-CDS position with polymorphism contributes 1 seg site."""
    ing = _seq_set({
        "Bgland_1_1": "ATGA",
        "Bgland_1_2": "ATGT",
    })
    out = _seq_set({
        "Bcrena_01_1": "ATGA",
    })
    ann = LocusAnnotation.empty(4, strand="+")
    for i in range(3):
        _annotate(ann, i, "CDS", codon_index=0, codon_pos=i)
    _annotate(ann, 3, "intergenic")

    seg = count_segregating_silent_annotated(ing, out, ann)
    # CDS codon ATG: 0 silent sites (Met), so 0 syn seg sites.
    # Intergenic pos 3: A/T segregating -> 1.
    assert seg == 1
