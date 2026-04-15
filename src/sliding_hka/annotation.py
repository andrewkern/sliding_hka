"""Per-position locus annotations: feature labels and codon structure.

A LocusAnnotation pairs each alignment column with:
  - a feature label (e.g. ``"CDS"``, ``"intron"``, ``"intergenic"``, ``""``),
  - for CDS positions, the codon index and the column-within-codon
    (0, 1, or 2). Non-CDS columns carry ``codon_index = -1``,
    ``codon_pos = -1``.

Codon-aware analysis can therefore stitch together codons whose three
positions are not contiguous in the alignment (e.g. when a codon spans
an intron in genomic space): the same ``codon_index`` is shared across
the three positions wherever they live.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LocusAnnotation:
    """Per-position annotation aligned to a multi-sequence FASTA."""

    n_positions: int
    feature: np.ndarray
    codon_index: np.ndarray
    codon_pos: np.ndarray
    strand: str = "+"
    """Strand of the CDS relative to the aligned (+-strand) sequences.
    For ``"-"``, codon bases read from the alignment are reverse-complemented
    before codon-aware classification."""

    def __post_init__(self) -> None:
        for arr_name in ("feature", "codon_index", "codon_pos"):
            arr = getattr(self, arr_name)
            if len(arr) != self.n_positions:
                raise ValueError(
                    f"{arr_name} length {len(arr)} != n_positions {self.n_positions}"
                )
        if self.strand not in ("+", "-"):
            raise ValueError(f"strand must be '+' or '-', got {self.strand!r}")

    @classmethod
    def empty(cls, n_positions: int, strand: str = "+") -> LocusAnnotation:
        return cls(
            n_positions=n_positions,
            feature=np.array([""] * n_positions, dtype="U16"),
            codon_index=np.full(n_positions, -1, dtype=np.int64),
            codon_pos=np.full(n_positions, -1, dtype=np.int8),
            strand=strand,
        )

    @classmethod
    def from_tsv(cls, path: str | Path, alignment_length: int, strand: str = "+") -> LocusAnnotation:
        """Load annotation from a TSV with columns
        ``pos_1based, feature, codon_index, codon_pos``.

        Positions present in the TSV override the empty default for that
        index; missing positions keep the empty annotation. The TSV may
        contain additional columns (they are ignored).
        """
        # Mutable views (frozen dataclass on Python's side, numpy arrays still mutate)
        n_seen = 0
        path = Path(path)
        # Optional `# strand=...` and `# foo=bar` header lines may precede the TSV body.
        body_lines: list[str] = []
        embedded_strand: str | None = None
        with path.open() as fh:
            for line in fh:
                if line.startswith("#"):
                    if "strand=" in line:
                        v = line.split("strand=", 1)[1].strip()
                        if v in ("+", "-"):
                            embedded_strand = v
                    continue
                body_lines.append(line)
        # Embedded header takes precedence only when caller supplied the default.
        if strand == "+" and embedded_strand is not None:
            strand = embedded_strand

        ann = cls.empty(alignment_length, strand=strand)
        feat = ann.feature
        cidx = ann.codon_index
        cpos = ann.codon_pos

        from io import StringIO
        reader = csv.DictReader(StringIO("".join(body_lines)), delimiter="\t")
        required = {"pos_1based", "feature", "codon_index", "codon_pos"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"TSV {path} missing columns: {sorted(missing)}")
        for row in reader:
            pos = int(row["pos_1based"]) - 1
            if pos < 0 or pos >= alignment_length:
                raise ValueError(
                    f"position {pos + 1} in {path} out of range "
                    f"[1, {alignment_length}]"
                )
            feat[pos] = row["feature"]
            cidx[pos] = int(row["codon_index"])
            cpos[pos] = int(row["codon_pos"])
            n_seen += 1
        if n_seen != alignment_length:
            raise ValueError(
                f"TSV {path} has {n_seen} rows, expected length {alignment_length}"
            )
        return ann

    def is_cds(self) -> np.ndarray:
        """Boolean mask of length n_positions; True at CDS positions."""
        return self.codon_index >= 0

    def codon_groups(self) -> dict[int, tuple[int, ...]]:
        """Return ``{codon_index: (pos_a, pos_b, pos_c)}`` sorted by codon_pos.

        Positions in each tuple are alignment indices in transcription
        order (codon_pos 0, 1, 2). Codons that span non-CDS columns
        (e.g. introns) still group their three positions correctly because
        the lookup is by ``codon_index``.
        """
        groups: dict[int, list[tuple[int, int]]] = {}
        for pos in range(self.n_positions):
            ci = int(self.codon_index[pos])
            if ci < 0:
                continue
            groups.setdefault(ci, []).append((int(self.codon_pos[pos]), pos))
        out: dict[int, tuple[int, ...]] = {}
        for ci, items in groups.items():
            items.sort(key=lambda x: x[0])
            out[ci] = tuple(p for _, p in items)
        return out
