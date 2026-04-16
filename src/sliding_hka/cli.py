"""Command-line interface for the sliding-window HKA diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import typer

from sliding_hka.annotation import LocusAnnotation
from sliding_hka.classic_hka import HKALocusInput, hka_test
from sliding_hka.counts import (
    count_segregating_silent,
    per_codon_arrays,
    per_position_arrays,
)
from sliding_hka.hka import LocusTotals, estimate_t_plus_1
from sliding_hka.io import load_msa
from sliding_hka.plot import sliding_hka_plot
from sliding_hka.window import sliding_window

app = typer.Typer(
    name="sliding-hka",
    help="Sliding-window HKA diagnostic for aligned coding sequences.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Sliding-window HKA diagnostic for aligned coding sequences."""


@app.command()
def run(
    fastas: list[Path] = typer.Argument(
        ..., help="One or more aligned CDS FASTAs.", exists=True, readable=True
    ),
    outdir: Path = typer.Option(
        Path("."), "--outdir", "-o", help="Directory for output PNGs."
    ),
    window: int = typer.Option(100, "--window", "-w", help="Window width in silent sites."),
    ingroup_match: str = typer.Option(
        None,
        "--ingroup-match",
        help=(
            "Substring matched against sequence names to select ingroup. "
            "If omitted, ingroup is everything not in the outgroup."
        ),
    ),
    outgroup_match: str = typer.Option(
        None,
        "--outgroup-match",
        help=(
            "Substring matched against sequence names to select outgroup. "
            "If omitted, the first sequence in each FASTA is used as the outgroup."
        ),
    ),
    allow_multi_outgroup: bool = typer.Option(
        False,
        "--allow-multi-outgroup",
        help="Permit more than one outgroup sequence per locus.",
    ),
    joint_t: bool = typer.Option(
        False, "--joint-t", help="Pool T+1 across all input loci."
    ),
    annotation_dir: Path = typer.Option(
        None,
        "--annotation-dir",
        help=(
            "Directory of per-locus annotation TSVs (one per FASTA, named "
            "<locus>.annotation.tsv). When provided, sliding HKA runs in "
            "per-position mode: CDS columns get codon-aware silent-site "
            "counts and non-CDS columns are treated as one silent site each."
        ),
    ),
    image_format: str = typer.Option("png", "--format", help="Image format (png or pdf)."),
) -> None:
    """Compute and plot observed vs. expected silent diversity along each CDS."""
    outdir.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[Path, tuple, np.ndarray | None, LocusAnnotation | None]] = []
    for fa in fastas:
        typer.echo(f"Loading {fa}", err=True)
        ingroup, outgroup = load_msa(
            fa,
            ingroup_match=ingroup_match,
            outgroup_match=outgroup_match,
            allow_multi_outgroup=allow_multi_outgroup,
        )
        sites, pi, div, nt_positions, ann = _load_arrays(fa, ingroup, outgroup, annotation_dir)
        loaded.append((fa, (sites, pi, div), nt_positions, ann))

    if joint_t:
        totals = [
            LocusTotals.from_arrays(s, p, d) for _, (s, p, d), _, _ in loaded
        ]
        t_plus_1 = estimate_t_plus_1(totals)
        typer.echo(f"Joint T+1 = {t_plus_1:.3f}", err=True)

    for fa, (sites, pi, div), nt_positions, ann in loaded:
        if not joint_t:
            t_plus_1 = estimate_t_plus_1(
                [LocusTotals.from_arrays(sites, pi, div)]
            )
            typer.echo(f"{fa.name}: T+1 = {t_plus_1:.3f}", err=True)

        out = sliding_window(
            sites, pi, div, t_plus_1=t_plus_1, w=window, nt_positions=nt_positions
        )
        save_path = outdir / f"{_locus_name(fa)}.sliding_hka.{image_format}"
        fig = sliding_hka_plot(
            out, t_plus_1=t_plus_1, window=window, locus=_locus_name(fa),
            save_to=save_path, annotation=ann,
        )
        plt.close(fig)
        typer.echo(f"Wrote {save_path}", err=True)


@app.command()
def test(
    fastas: list[Path] = typer.Argument(
        ..., help="Two or more aligned FASTAs (one per locus).", exists=True, readable=True
    ),
    ingroup_match: str = typer.Option(None, "--ingroup-match"),
    outgroup_match: str = typer.Option(None, "--outgroup-match"),
    allow_multi_outgroup: bool = typer.Option(False, "--allow-multi-outgroup"),
    mode: str = typer.Option(
        "pwd", "--mode", help="Polymorphism measure: 'pwd' (pairwise diffs) or 'seg' (segregating sites)."
    ),
    annotation_dir: Path = typer.Option(None, "--annotation-dir"),
) -> None:
    """Run the classic HKA test (Hudson, Kreitman & Aguade 1987).

    Requires at least two loci. Tests whether the ratio of within-species
    polymorphism to between-species divergence is homogeneous across loci
    under a constant-rate neutral model.

    Reports the chi-squared statistic, p-value, and a per-locus post-hoc
    table showing direction of deviation (excess polymorphism = balancing
    selection candidate; deficit = sweep/constraint candidate).
    """
    locus_inputs: list[HKALocusInput] = []
    for fa in fastas:
        typer.echo(f"Loading {fa.name}", err=True)
        ingroup, outgroup = load_msa(
            fa,
            ingroup_match=ingroup_match,
            outgroup_match=outgroup_match,
            allow_multi_outgroup=allow_multi_outgroup,
        )
        sites, pi_arr, div_arr, _, _ = _load_arrays(fa, ingroup, outgroup, annotation_dir)

        if mode == "seg":
            poly_val = float(count_segregating_silent(ingroup, outgroup))
        else:
            poly_val = float(pi_arr.sum())

        locus_inputs.append(
            HKALocusInput(
                locus=_locus_name(fa),
                poly=poly_val,
                div=float(div_arr.sum()),
                n_seqs=len(ingroup),
            )
        )

    result = hka_test(locus_inputs, mode=mode)

    typer.echo("")
    typer.echo(f"Classic HKA Test ({mode} mode)")
    typer.echo("=" * 40)
    typer.echo(f"T + 1 = {result.t_hat + 1:.3f}")
    typer.echo(f"X^2   = {result.chi2:.4f}  (df = {result.df}, p = {result.p_value:.4f})")
    typer.echo("")

    hdr = f"{'Locus':<8} {'obs_poly':>10} {'exp_poly':>10} {'obs_div':>10} {'exp_div':>10} {'chi2':>8} {'direction'}"
    typer.echo(hdr)
    typer.echo("-" * len(hdr))
    for r in result.per_locus:
        typer.echo(
            f"{r.locus:<8} {r.obs_poly:>10.2f} {r.exp_poly:>10.2f} "
            f"{r.obs_div:>10.2f} {r.exp_div:>10.2f} {r.chi2_poly + r.chi2_div:>8.3f} "
            f"{r.direction}"
        )


def _locus_name(fa: Path) -> str:
    """Derive a clean locus name from a FASTA path (strip .full suffix)."""
    name = fa.stem
    if name.endswith(".full"):
        name = name[:-5]
    return name


def _load_arrays(
    fa: Path,
    ingroup,
    outgroup,
    annotation_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, LocusAnnotation | None]:
    """Load per-codon or per-position arrays, with optional annotation.

    Returns (sites, pi, div, nt_positions_or_None, annotation_or_None).
    """
    locus_name = _locus_name(fa)
    if annotation_dir is not None:
        ann_path = annotation_dir / f"{locus_name}.annotation.tsv"
        if ann_path.exists():
            ann = LocusAnnotation.from_tsv(
                ann_path,
                alignment_length=ingroup.alignment_length,
                strand=_infer_strand(ann_path),
            )
            sites, pi, div = per_position_arrays(ingroup, outgroup, ann)
            nt_positions = np.arange(1, ann.n_positions + 1, dtype=int)
            return sites, pi, div, nt_positions, ann
        typer.echo(
            f"  warning: missing annotation {ann_path}, falling back to per-codon",
            err=True,
        )
    sites, pi, div = per_codon_arrays(ingroup, outgroup)
    return sites, pi, div, None, None


def _infer_strand(annotation_tsv: Path) -> str:
    """Read a `# strand=...` header comment from the annotation TSV, if present."""
    with annotation_tsv.open() as fh:
        for line in fh:
            if line.startswith("# strand="):
                value = line.split("=", 1)[1].strip()
                if value in ("+", "-"):
                    return value
            if not line.startswith("#"):
                break
    return "+"
