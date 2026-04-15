"""Tests for sliding_hka.cli."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from typer.testing import CliRunner

from sliding_hka.cli import app

FIXTURES = Path(__file__).parent / "data"

runner = CliRunner()


def test_cli_help_runs_and_describes_tool():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sliding-window HKA" in result.stdout.lower() or "sliding" in result.stdout.lower()


def test_cli_runs_on_synthetic_fasta(tmp_path: Path):
    result = runner.invoke(
        app,
        ["run", str(FIXTURES / "synthetic_min.fa"), "--outdir", str(tmp_path), "--window", "1"],
    )
    assert result.exit_code == 0, result.output
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0


def test_cli_joint_t_emits_one_plot_per_locus(tmp_path: Path):
    # Build a second fixture in tmp_path so we have two loci
    alt = tmp_path / "alt.fa"
    alt.write_text((FIXTURES / "synthetic_min.fa").read_text())

    result = runner.invoke(
        app,
        [
            "run",
            str(FIXTURES / "synthetic_min.fa"),
            str(alt),
            "--outdir",
            str(tmp_path),
            "--window",
            "1",
            "--joint-t",
        ],
    )
    assert result.exit_code == 0, result.output
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 2


def test_cli_errors_on_missing_file(tmp_path: Path):
    result = runner.invoke(
        app,
        ["run", str(tmp_path / "nope.fa"), "--outdir", str(tmp_path)],
    )
    assert result.exit_code != 0
