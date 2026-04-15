"""Load aligned coding-sequence MSAs and split into ingroup / outgroup."""

from __future__ import annotations

from pathlib import Path

from mkado.core.sequences import Sequence, SequenceSet
from mkado.io.fasta import read_fasta


def load_msa(
    path: str | Path,
    *,
    ingroup_prefix: str = "Bgland_",
    outgroup_prefix: str = "Bcrena_",
    allow_multi_outgroup: bool = False,
) -> tuple[SequenceSet, SequenceSet]:
    """Load a CDS MSA and split sequences by name prefix.

    Args:
        path: Path to aligned FASTA.
        ingroup_prefix: Prefix identifying ingroup sequences.
        outgroup_prefix: Prefix identifying outgroup sequences.
        allow_multi_outgroup: If False (default), having more than one
            outgroup sequence is an error.

    Returns:
        Tuple of (ingroup_set, outgroup_set).

    Raises:
        ValueError: if no outgroup sequences found, if multiple outgroups
            found when ``allow_multi_outgroup`` is False, if sequences
            have mismatched lengths, or if alignment length is not a
            multiple of three.
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

    ingroup_seqs: list[Sequence] = []
    outgroup_seqs: list[Sequence] = []
    for name, seq in records:
        if name.startswith(outgroup_prefix):
            outgroup_seqs.append(Sequence(name=name, sequence=seq))
        elif name.startswith(ingroup_prefix):
            ingroup_seqs.append(Sequence(name=name, sequence=seq))

    if not outgroup_seqs:
        raise ValueError(
            f"No outgroup sequences (prefix {outgroup_prefix!r}) in {path}"
        )
    if len(outgroup_seqs) > 1 and not allow_multi_outgroup:
        raise ValueError(
            f"Found multiple outgroup sequences in {path}; "
            "pass allow_multi_outgroup=True to permit this."
        )
    if not ingroup_seqs:
        raise ValueError(
            f"No ingroup sequences (prefix {ingroup_prefix!r}) in {path}"
        )

    return SequenceSet(sequences=ingroup_seqs), SequenceSet(sequences=outgroup_seqs)
