# sliding_hka

Sliding-window HKA diagnostic for aligned coding sequences, following Kreitman & Hudson (1991, *Genetics* 127:565–582).

Given a CDS multiple-sequence alignment with one outgroup sequence and several ingroup sequences, computes per-codon silent polymorphism and silent divergence, then plots a sliding window of observed pairwise silent diversity (π) against the divergence-scaled neutral expectation. Peaks suggest balanced polymorphism; troughs suggest sweeps or constraint.

## Install

```bash
uv sync
```

## Usage

```bash
uv run sliding-hka run msas/Adh.fa --outdir out/
uv run sliding-hka run msas/*.fa --outdir out/ --joint-t
```

Default sequence-name prefixes: `Bcrena_` for outgroup, `Bgland_` for ingroup. Override with `--outgroup-prefix` / `--ingroup-prefix`.

## Notes on method

- **Silent sites** per codon use Nei–Gojobori on the outgroup codon.
- **Divergence** per codon is averaged over all ingroup × outgroup pairwise comparisons (the 1991 paper uses one allele; averaging is more stable).
- **T+1** is estimated as `silent divergence / silent π`; with `--joint-t`, totals are pooled across all input loci.
- Window width is measured in **silent sites** (default 100), not base pairs.
