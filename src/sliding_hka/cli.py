"""Command-line interface for the sliding-window HKA diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import typer

from sliding_hka.counts import per_codon_arrays
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
    image_format: str = typer.Option("png", "--format", help="Image format (png or pdf)."),
) -> None:
    """Compute and plot observed vs. expected silent diversity along each CDS."""
    outdir.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[Path, tuple]] = []
    for fa in fastas:
        typer.echo(f"Loading {fa}", err=True)
        ingroup, outgroup = load_msa(
            fa,
            ingroup_match=ingroup_match,
            outgroup_match=outgroup_match,
            allow_multi_outgroup=allow_multi_outgroup,
        )
        sites, pi, div = per_codon_arrays(ingroup, outgroup)
        loaded.append((fa, (sites, pi, div)))

    if joint_t:
        totals = [
            LocusTotals.from_arrays(s, p, d) for _, (s, p, d) in loaded
        ]
        t_plus_1 = estimate_t_plus_1(totals)
        typer.echo(f"Joint T+1 = {t_plus_1:.3f}", err=True)

    for fa, (sites, pi, div) in loaded:
        if not joint_t:
            t_plus_1 = estimate_t_plus_1(
                [LocusTotals.from_arrays(sites, pi, div)]
            )
            typer.echo(f"{fa.name}: T+1 = {t_plus_1:.3f}", err=True)

        out = sliding_window(sites, pi, div, t_plus_1=t_plus_1, w=window)
        save_path = outdir / f"{fa.stem}.sliding_hka.{image_format}"
        fig = sliding_hka_plot(
            out, t_plus_1=t_plus_1, window=window, locus=fa.stem, save_to=save_path
        )
        plt.close(fig)
        typer.echo(f"Wrote {save_path}", err=True)
