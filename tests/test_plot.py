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
    title = fig.axes[0].get_title()
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
