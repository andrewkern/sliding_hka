"""Tests for sliding_hka.plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.figure

from sliding_hka.plot import sliding_hka_plot


def _fake_window_output(n: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "nt_position": np.arange(n) * 3 + 1,
        "obs_pi": rng.uniform(0.0, 0.05, size=n),
        "exp_pi": rng.uniform(0.0, 0.05, size=n),
        "sites_in_window": np.full(n, 100.0),
    }


def test_plot_returns_figure():
    out = _fake_window_output()
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_writes_png(tmp_path: Path):
    out = _fake_window_output()
    path = tmp_path / "out.png"
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh", save_to=path)
    assert path.exists()
    assert path.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_title_mentions_t_plus_1_and_window():
    out = _fake_window_output()
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh")
    title = fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title()
    assert "T+1" in title
    assert "6.3" in title or "6.30" in title
    assert "w=100" in title
    assert "Adh" in title
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_skips_nan_positions_without_crashing():
    out = _fake_window_output()
    out["obs_pi"][:5] = np.nan
    out["exp_pi"][-5:] = np.nan
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh")
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_with_annotation_adds_structure_axis():
    from sliding_hka.annotation import LocusAnnotation
    out = _fake_window_output(n=30)
    ann = LocusAnnotation.empty(90, strand="+")
    for p in range(0, 30):
        ann.feature[p] = "intergenic"
    for p in range(30, 60):
        ann.feature[p] = "CDS"
        ann.codon_index[p] = (p - 30) // 3
        ann.codon_pos[p] = (p - 30) % 3
    for p in range(60, 90):
        ann.feature[p] = "intron"
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh", annotation=ann)
    assert len(fig.axes) == 2  # main + gene-structure track
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_short_gaps_render_as_single_panel():
    # NaN run of 10 positions × 3 bp apart = 30bp: stays below the default
    # 2000bp max_gap_bp, so the plot renders as a single axes.
    out = _fake_window_output(n=30)
    out["obs_pi"][10:20] = np.nan
    out["exp_pi"][10:20] = np.nan
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh")
    assert len(fig.axes) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_long_gap_triggers_broken_axis():
    # Fake a very long NaN gap by explicitly setting widely-spaced positions.
    n = 30
    pos = np.arange(n) * 1000 + 1  # 1000bp apart -> total 29kb
    out = {
        "nt_position": pos,
        "obs_pi": np.linspace(0.01, 0.05, n),
        "exp_pi": np.linspace(0.02, 0.04, n),
        "sites_in_window": np.full(n, 100.0),
    }
    # Punch 10 consecutive NaN positions = 10,000bp gap in the middle
    out["obs_pi"][10:20] = np.nan
    out["exp_pi"][10:20] = np.nan
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh",
                           max_gap_bp=2000)
    # Broken axis → two main panels (no annotation, so no track row)
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_long_gap_with_annotation_breaks_both_tracks():
    from sliding_hka.annotation import LocusAnnotation
    n = 30
    pos = np.arange(n) * 1000 + 1
    out = {
        "nt_position": pos,
        "obs_pi": np.linspace(0.01, 0.05, n),
        "exp_pi": np.linspace(0.02, 0.04, n),
        "sites_in_window": np.full(n, 100.0),
    }
    out["obs_pi"][10:20] = np.nan
    out["exp_pi"][10:20] = np.nan
    ann = LocusAnnotation.empty(30000, strand="+")
    for p in range(5000, 8000):
        ann.feature[p] = "CDS"
        ann.codon_index[p] = (p - 5000) // 3
        ann.codon_pos[p] = (p - 5000) % 3
    fig = sliding_hka_plot(out, t_plus_1=6.3, window=100, locus="Adh",
                           annotation=ann, max_gap_bp=2000)
    # 2 main panels × 2 rows (main + track) = 4 axes
    assert len(fig.axes) == 4
    import matplotlib.pyplot as plt
    plt.close(fig)
