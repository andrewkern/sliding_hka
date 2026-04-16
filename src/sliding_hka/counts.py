"""Per-codon and per-position silent-site, silent-pi, and silent-divergence counts."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from mkado.core.codons import DEFAULT_CODE, GeneticCode
from mkado.core.sequences import SequenceSet

from sliding_hka.annotation import LocusAnnotation


_COMPLEMENT = str.maketrans("ACGTNacgtn-", "TGCANtgcan-")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _is_clean(codon: str) -> bool:
    return len(codon) == 3 and "N" not in codon and "-" not in codon


def silent_sites_codon(codon: str, code: GeneticCode = DEFAULT_CODE) -> float:
    """Nei-Gojobori synonymous-site count for one codon; 0 if codon is dirty."""
    if not _is_clean(codon):
        return 0.0
    return code.count_synonymous_sites(codon)


def _silent_diffs_between(
    codon_a: str, codon_b: str, code: GeneticCode
) -> int | None:
    """Count synonymous changes along the shortest path, or None if no path."""
    if codon_a == codon_b:
        return 0
    path = code.get_path(codon_a, codon_b)
    if not path:
        return None
    return sum(1 for change_type, _ in path if change_type == "S")


def segregating_silent_codon(
    codons: list[str], code: GeneticCode = DEFAULT_CODE
) -> int:
    """Count of synonymous segregating sites at this codon position.

    A codon position is a segregating silent site if at least one pair of
    clean ingroup codons differs by a synonymous change. Returns 0 or 1.
    """
    clean = [c for c in codons if _is_clean(c)]
    if len(clean) < 2:
        return 0
    for a, b in combinations(clean, 2):
        s = _silent_diffs_between(a, b, code)
        if s is not None and s > 0:
            return 1
    return 0


def count_segregating_silent(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
    code: GeneticCode = DEFAULT_CODE,
) -> int:
    """Total count of silent segregating sites across all codons.

    A codon contributes 1 if any ingroup pair differs by a synonymous
    change at that position and the outgroup codon is clean (so we have a
    valid silent-site denominator). Returns an integer count suitable for
    the Seg mode of the classic HKA test.
    """
    n_codons = ingroup.num_codons
    total = 0
    for c in range(n_codons):
        out_codon = _outgroup_representative_codon(outgroup, c)
        if out_codon is None:
            continue
        if silent_sites_codon(out_codon, code) <= 0:
            continue
        in_codons = [s.get_codon(c, ingroup.reading_frame) for s in ingroup.sequences]
        total += segregating_silent_codon(in_codons, code)
    return total


def count_segregating_silent_annotated(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
    annotation: LocusAnnotation,
    code: GeneticCode = DEFAULT_CODE,
) -> int:
    """Count silent segregating sites using per-position annotation.

    CDS positions: a codon contributes 1 if any ingroup pair differs by a
    synonymous change (same logic as ``count_segregating_silent``).

    Non-CDS positions: a site contributes 1 if at least two distinct clean
    bases exist among the ingroup sequences at that column and the outgroup
    is also clean (so the position is alignable).
    """
    total = 0
    is_cds = annotation.is_cds()

    # Non-CDS: nucleotide-level segregation
    for col in range(annotation.n_positions):
        if is_cds[col]:
            continue
        out_bases = [s.sequence[col].upper() for s in outgroup.sequences]
        if not any(b in "ACGT" for b in out_bases):
            continue
        in_bases = [s.sequence[col].upper() for s in ingroup.sequences]
        clean = {b for b in in_bases if b in "ACGT"}
        if len(clean) >= 2:
            total += 1

    # CDS: codon-aware synonymous segregation
    for _codon_idx, positions in annotation.codon_groups().items():
        if len(positions) != 3:
            continue
        out_codons = [
            _read_codon(seq.sequence, positions, annotation.strand)
            for seq in outgroup.sequences
        ]
        out_repr = _representative_codon(out_codons)
        if out_repr is None:
            continue
        if silent_sites_codon(out_repr, code) <= 0:
            continue
        ing_codons = [
            _read_codon(seq.sequence, positions, annotation.strand)
            for seq in ingroup.sequences
        ]
        total += segregating_silent_codon(ing_codons, code)

    return total


def per_site_arrays_all(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-site arrays treating every aligned position as one site.

    No codon awareness -- counts all nucleotide differences (synonymous
    and replacement alike). Useful when you want the HKA test to operate
    on total variation rather than silent variation only.

    Returns three numpy arrays of length ``alignment_length``:
    ``sites``, ``pi``, ``div``.
    """
    n = ingroup.alignment_length
    sites = np.zeros(n)
    pi = np.zeros(n)
    div = np.zeros(n)
    for col in range(n):
        s, p, d = _per_site_at(col, ingroup, outgroup)
        sites[col] = s
        pi[col] = p
        div[col] = d
    return sites, pi, div


def count_segregating_all(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
) -> int:
    """Count all segregating sites (any position with 2+ ingroup alleles).

    Unlike ``count_segregating_silent`` this includes replacement changes.
    Only positions where the outgroup has a clean base are counted (so the
    site is alignable for divergence comparison).
    """
    total = 0
    for col in range(ingroup.alignment_length):
        out_bases = [s.sequence[col].upper() for s in outgroup.sequences]
        if not any(b in "ACGT" for b in out_bases):
            continue
        in_bases = [s.sequence[col].upper() for s in ingroup.sequences]
        clean = {b for b in in_bases if b in "ACGT"}
        if len(clean) >= 2:
            total += 1
    return total


def silent_pairwise_diff_codon(
    codons: list[str], code: GeneticCode = DEFAULT_CODE
) -> tuple[float, int]:
    """Mean silent differences across all clean-codon pairs at a position.

    Args:
        codons: One codon per sequence (may include gapped/N codons).
        code: Genetic code for path lookup.

    Returns:
        Tuple ``(mean_silent_diffs, n_pairs)``. Returns ``(nan, 0)`` when
        fewer than two clean codons are available.
    """
    clean = [c for c in codons if _is_clean(c)]
    total_s = 0
    n_pairs = 0
    for a, b in combinations(clean, 2):
        s = _silent_diffs_between(a, b, code)
        if s is None:
            continue
        total_s += s
        n_pairs += 1
    if n_pairs == 0:
        return math.nan, 0
    return total_s / n_pairs, n_pairs


def silent_divergence_codon(
    ingroup_codons: list[str],
    outgroup_codon: str,
    code: GeneticCode = DEFAULT_CODE,
) -> tuple[float, int]:
    """Mean silent differences between each ingroup codon and the outgroup.

    Args:
        ingroup_codons: One codon per ingroup sequence (may be dirty).
        outgroup_codon: Single outgroup codon.
        code: Genetic code for path lookup.

    Returns:
        Tuple ``(mean_silent_diffs, n_obs)``. Returns ``(nan, 0)`` when
        the outgroup codon is dirty or no clean ingroup codons remain.
    """
    if not _is_clean(outgroup_codon):
        return math.nan, 0
    total_s = 0
    n_obs = 0
    for c in ingroup_codons:
        if not _is_clean(c):
            continue
        s = _silent_diffs_between(c, outgroup_codon, code)
        if s is None:
            continue
        total_s += s
        n_obs += 1
    if n_obs == 0:
        return math.nan, 0
    return total_s / n_obs, n_obs


def per_codon_arrays(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
    code: GeneticCode = DEFAULT_CODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-codon silent-site, silent-pi, and silent-divergence arrays.

    Codons where the outgroup is dirty (or no clean ingroup codons remain,
    or no pairs can be formed) contribute zeros — so they are neutral in
    downstream sums but preserve their codon index for positional alignment.

    Args:
        ingroup: Ingroup sequences.
        outgroup: Outgroup sequences. If more than one, the consensus codon
            (majority vote over clean codons) is used per position; ties
            or positions with no clean outgroup codons are treated as dirty.
        code: Genetic code.

    Returns:
        Three numpy arrays of length ``n_codons``:
        ``silent_sites``, ``silent_pi``, ``silent_div``.
    """
    n_codons = ingroup.num_codons
    if outgroup.num_codons != n_codons:
        raise ValueError(
            f"codon count mismatch: ingroup={n_codons}, outgroup={outgroup.num_codons}"
        )
    if ingroup.alignment_length % 3 != 0:
        raise ValueError(
            f"per_codon_arrays requires alignment length divisible by 3; "
            f"got {ingroup.alignment_length}"
        )

    sites = np.zeros(n_codons)
    pi = np.zeros(n_codons)
    div = np.zeros(n_codons)

    for c in range(n_codons):
        out_codon = _outgroup_representative_codon(outgroup, c)
        if out_codon is None:
            continue
        in_codons = [s.get_codon(c, ingroup.reading_frame) for s in ingroup.sequences]

        s_val = silent_sites_codon(out_codon, code)
        if s_val <= 0:
            continue
        pi_val, n_pairs = silent_pairwise_diff_codon(in_codons, code)
        div_val, n_div = silent_divergence_codon(in_codons, out_codon, code)

        sites[c] = s_val
        if n_pairs > 0:
            pi[c] = pi_val
        if n_div > 0:
            div[c] = div_val

    return sites, pi, div


def per_position_arrays(
    ingroup: SequenceSet,
    outgroup: SequenceSet,
    annotation: LocusAnnotation,
    code: GeneticCode = DEFAULT_CODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position silent_sites, silent_pi, silent_div arrays for an
    annotated multi-feature alignment.

    CDS positions get their codon-level Nei-Gojobori silent-site count and
    pairwise/divergence silent-change counts (computed as in
    ``per_codon_arrays``), then those scalars are spread evenly across the
    codon's three alignment positions. Codons whose three positions are not
    contiguous (e.g. a codon spanning an intron boundary) are still grouped
    correctly by ``annotation.codon_index``.

    Non-CDS positions are treated per-site: every alignable position
    contributes 1 silent site; pi and div are mean nucleotide-level
    pairwise differences.

    Args:
        ingroup: Aligned ingroup sequences.
        outgroup: Aligned outgroup sequences (one or more).
        annotation: Per-position feature labels and codon structure.
        code: Genetic code (used for CDS classification).

    Returns:
        Three numpy arrays of length ``annotation.n_positions``:
        silent_sites, silent_pi, silent_div.
    """
    n = annotation.n_positions
    if ingroup.alignment_length < n or outgroup.alignment_length < n:
        raise ValueError(
            f"alignment length {min(ingroup.alignment_length, outgroup.alignment_length)} "
            f"< annotation length {n}"
        )

    sites = np.zeros(n)
    pi = np.zeros(n)
    div = np.zeros(n)

    # --- Non-CDS positions: per-site nt diversity ---
    is_cds = annotation.is_cds()
    for col in range(n):
        if is_cds[col]:
            continue
        s, p, d = _per_site_at(col, ingroup, outgroup)
        sites[col] = s
        pi[col] = p
        div[col] = d

    # --- CDS codons: codon-aware classification, distributed across the 3 positions ---
    for codon_idx, positions in annotation.codon_groups().items():
        if len(positions) != 3:
            # malformed annotation; skip with zeros
            continue
        ing_codons = [
            _read_codon(seq.sequence, positions, annotation.strand)
            for seq in ingroup.sequences
        ]
        out_codons = [
            _read_codon(seq.sequence, positions, annotation.strand)
            for seq in outgroup.sequences
        ]
        out_repr = _representative_codon(out_codons)
        if out_repr is None:
            continue

        s_codon = silent_sites_codon(out_repr, code)
        if s_codon <= 0:
            continue
        pi_codon, n_pairs = silent_pairwise_diff_codon(ing_codons, code)
        div_codon, n_div = silent_divergence_codon(ing_codons, out_repr, code)

        third_s = s_codon / 3.0
        third_pi = (pi_codon / 3.0) if n_pairs > 0 else 0.0
        third_div = (div_codon / 3.0) if n_div > 0 else 0.0
        for p_idx in positions:
            sites[p_idx] = third_s
            pi[p_idx] = third_pi
            div[p_idx] = third_div

    return sites, pi, div


def _read_codon(seq: str, positions: tuple[int, ...], strand: str) -> str:
    """Read 3 bases from `seq` at the given alignment positions, applying
    reverse-complement when ``strand == '-'``."""
    bases = seq[positions[0]] + seq[positions[1]] + seq[positions[2]]
    if strand == "-":
        bases = _revcomp(bases)
    return bases.upper()


def _representative_codon(codons: list[str]) -> str | None:
    """Return the unique clean codon if all clean codons agree, else None."""
    clean = [c for c in codons if _is_clean(c)]
    if not clean:
        return None
    unique = set(clean)
    if len(unique) == 1:
        return clean[0]
    return None


def _per_site_at(
    col: int, ingroup: SequenceSet, outgroup: SequenceSet
) -> tuple[float, float, float]:
    """Per-site silent_sites/pi/div for a single (assumed neutral) column.

    Returns (1.0, pi, div) when both ingroup and outgroup have any clean
    bases at this column, else (0, 0, 0). Pairs/observations involving N
    or '-' are dropped.
    """
    in_bases = [s.sequence[col].upper() for s in ingroup.sequences]
    out_bases = [s.sequence[col].upper() for s in outgroup.sequences]
    in_clean = [b for b in in_bases if b in "ACGT"]
    out_clean = [b for b in out_bases if b in "ACGT"]
    if not in_clean or not out_clean:
        return 0.0, 0.0, 0.0

    pi_val = 0.0
    if len(in_clean) >= 2:
        n_pairs = 0
        n_diff = 0
        for a, b in combinations(in_clean, 2):
            n_pairs += 1
            if a != b:
                n_diff += 1
        pi_val = n_diff / n_pairs

    n_pairs = 0
    n_diff = 0
    for a in in_clean:
        for b in out_clean:
            n_pairs += 1
            if a != b:
                n_diff += 1
    div_val = (n_diff / n_pairs) if n_pairs else 0.0

    return 1.0, pi_val, div_val


def _outgroup_representative_codon(outgroup: SequenceSet, codon_index: int) -> str | None:
    """Return a single representative outgroup codon, or None if none is clean.

    With one outgroup sequence, returns that codon if clean. With multiple,
    returns the unique clean codon if all agree, else None.
    """
    codons = [s.get_codon(codon_index, outgroup.reading_frame) for s in outgroup.sequences]
    clean = [c for c in codons if _is_clean(c)]
    if not clean:
        return None
    unique = set(clean)
    if len(unique) == 1:
        return clean[0]
    return None
