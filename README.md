# sliding_hka

HKA test toolkit for aligned coding sequences, implementing both the classic HKA test (Hudson, Kreitman & Aguade 1987, *Genetics* 116:153-159) and the sliding-window HKA diagnostic (Kreitman & Hudson 1991, *Genetics* 127:565-582).

## Install

```bash
uv sync
```

## Usage

### Classic HKA test

Test whether the ratio of within-species polymorphism to between-species divergence is homogeneous across loci. Reports the chi-squared statistic, p-value, and per-locus direction of deviation.

```bash
# Pairwise-difference mode (default)
uv run sliding-hka test msas/*.fa

# Segregating-sites mode (original 1987 formulation)
uv run sliding-hka test msas/*.fa --mode seg

# With explicit ingroup/outgroup routing
uv run sliding-hka test msas/*.fa --outgroup-match Bcrena_ --ingroup-match Bgland_
```

Example output:

```
Classic HKA Test (pwd mode)
========================================
T + 1 = 5.711
X^2   = 3.4377  (df = 6, p = 0.7522)

Locus      obs_poly   exp_poly    obs_div    exp_div     chi2 direction
-----------------------------------------------------------------------
Adh            9.76      10.33      59.58      59.01    0.012 deficit_poly
Gpi           26.42      19.28     103.00     110.13    0.580 excess_poly
...
```

The `direction` column indicates whether a locus has more polymorphism than expected from its divergence (`excess_poly`, consistent with balancing selection) or less (`deficit_poly`, consistent with a selective sweep or constraint).

### Sliding-window HKA diagnostic

Plot observed vs. expected silent pairwise diversity along each locus.

```bash
uv run sliding-hka run msas/Adh.fa --outdir out/
uv run sliding-hka run msas/*.fa --outdir out/ --joint-t
```

For full-locus alignments (CDS + intron + flank) with per-position annotations:

```bash
uv run sliding-hka run out/per_locus_aln/full_msa/*.full.fa \
    --outdir out/hka/full_joint --window 500 \
    --outgroup-match BalCre_ --ingroup-match Bgland_ \
    --annotation-dir out/per_locus_aln/annotation --joint-t
```

### Ingroup / outgroup routing

By default, the first sequence in each FASTA is treated as the outgroup and the rest as the ingroup. Override with substring patterns `--outgroup-match PATTERN` and/or `--ingroup-match PATTERN`. Pass `--allow-multi-outgroup` to permit more than one outgroup sequence per locus.

## Notes on method

- **Silent sites** per codon use Nei-Gojobori on the outgroup codon.
- **Divergence** per codon is averaged over all ingroup x outgroup pairwise comparisons (the 1991 paper uses one allele; averaging is more stable).
- **T+1** is estimated as `silent divergence / silent pi`; with `--joint-t`, totals are pooled across all input loci.
- Sliding-window width is measured in **silent sites** (default 100 for CDS-only, 500 recommended for full-locus), not base pairs.
- The classic HKA test supports both **Seg** (segregating-site count, original 1987) and **Pwd** (pairwise differences, 1991 extension) polymorphism measures.
- With `--annotation-dir`, CDS positions get codon-aware silent-site counting and non-CDS positions (intron, UTR, intergenic) are treated as fully silent, following the 1991 paper's convention.
