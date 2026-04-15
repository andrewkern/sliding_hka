"""Load aligned coding-sequence MSAs and split into ingroup / outgroup."""

from __future__ import annotations

from pathlib import Path

from mkado.core.sequences import Sequence, SequenceSet
from mkado.io.fasta import read_fasta


def load_msa(
    path: str | Path,
    *,
    ingroup_match: str | None = None,
    outgroup_match: str | None = None,
    allow_multi_outgroup: bool = False,
) -> tuple[SequenceSet, SequenceSet]:
    """Load a CDS MSA and split sequences into ingroup and outgroup.

    Routing rules, in priority order:

    1. If ``outgroup_match`` is given, sequences whose names contain it are
       assigned to the outgroup. ``ingroup_match`` (if given) restricts the
       ingroup to matching sequences; otherwise the ingroup is everything
       that didn't match the outgroup.
    2. If only ``ingroup_match`` is given, the ingroup contains matching
       sequences and the outgroup is everything else.
    3. With no match patterns (default), the first sequence in the FASTA
       is taken as the outgroup and the remaining sequences as the ingroup.

    Args:
        path: Path to aligned FASTA.
        ingroup_match: Substring matched against sequence names.
        outgroup_match: Substring matched against sequence names.
        allow_multi_outgroup: If False (default), the outgroup must contain
            exactly one sequence after routing.

    Returns:
        Tuple of (ingroup_set, outgroup_set).

    Raises:
        ValueError: on length mismatch, non-triplet alignment length,
            empty ingroup or outgroup, or multiple outgroup sequences
            (unless ``allow_multi_outgroup`` is True).
    """
    records = list(read_fasta(path))
    if not records:
        raise ValueError(f"No sequences in {path}")

    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise ValueError(
            f"Alignment length mismatch in {path}: lengths {sorted(lengths)}"
        )
    (length,) = lengths
    if length % 3 != 0:
        raise ValueError(
            f"Alignment length in {path} ({length}) is not a multiple of three"
        )

    ingroup_seqs, outgroup_seqs = _route_sequences(
        records,
        ingroup_match=ingroup_match,
        outgroup_match=outgroup_match,
    )

    if not outgroup_seqs:
        hint = (
            f"no sequences match outgroup pattern {outgroup_match!r}"
            if outgroup_match is not None
            else "no outgroup sequences"
        )
        raise ValueError(f"{hint} in {path}")
    if len(outgroup_seqs) > 1 and not allow_multi_outgroup:
        raise ValueError(
            f"Found {len(outgroup_seqs)} outgroup sequences in {path}; "
            "pass allow_multi_outgroup=True (or tighten --outgroup-match) to permit this."
        )
    if not ingroup_seqs:
        hint = (
            f"no sequences match ingroup pattern {ingroup_match!r}"
            if ingroup_match is not None
            else "no ingroup sequences"
        )
        raise ValueError(f"{hint} in {path}")

    return (
        SequenceSet(sequences=[Sequence(name=n, sequence=s) for n, s in ingroup_seqs]),
        SequenceSet(sequences=[Sequence(name=n, sequence=s) for n, s in outgroup_seqs]),
    )


def _route_sequences(
    records: list[tuple[str, str]],
    *,
    ingroup_match: str | None,
    outgroup_match: str | None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if outgroup_match is not None:
        outgroup = [(n, s) for n, s in records if outgroup_match in n]
        if ingroup_match is not None:
            ingroup = [(n, s) for n, s in records if ingroup_match in n]
        else:
            ingroup = [
                (n, s) for n, s in records if outgroup_match not in n
            ]
        return ingroup, outgroup

    if ingroup_match is not None:
        ingroup = [(n, s) for n, s in records if ingroup_match in n]
        outgroup = [(n, s) for n, s in records if ingroup_match not in n]
        return ingroup, outgroup

    # Default: first sequence is the outgroup, the rest are ingroup.
    return list(records[1:]), list(records[:1])
