"""Per-codon silent-site, silent-pi, and silent-divergence counts."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from mkado.core.codons import DEFAULT_CODE, GeneticCode
from mkado.core.sequences import SequenceSet


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
