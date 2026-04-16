"""Classic HKA test (Hudson, Kreitman & Aguade 1987).

Given L >= 2 loci with within-species polymorphism and between-species
divergence, tests whether the ratio of polymorphism to divergence is
homogeneous across loci under a constant-rate neutral model. A departure
indicates natural selection at one or more loci.

Supports two polymorphism measures:
  * **Seg** -- number of segregating sites (original 1987 formulation).
  * **Pwd** -- average pairwise nucleotide differences (1991 extension).

Both use the same divergence measure (D_i = between-species differences).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from scipy.stats import chi2


@dataclass(frozen=True)
class HKALocusInput:
    """Per-locus summary statistics fed into the HKA test."""

    locus: str
    poly: float    # S_i (Seg) or pi_i (Pwd)
    div: float     # D_i (between-species differences)
    n_seqs: int    # number of ingroup sequences


@dataclass(frozen=True)
class HKALocusResult:
    """Per-locus output of the HKA test."""

    locus: str
    theta_hat: float
    obs_poly: float
    exp_poly: float
    var_poly: float
    obs_div: float
    exp_div: float
    var_div: float
    chi2_poly: float
    chi2_div: float
    direction: str  # "excess_poly" | "deficit_poly"


@dataclass(frozen=True)
class HKATestResult:
    """Aggregate result of the classic HKA test."""

    mode: str
    chi2: float
    df: int
    p_value: float
    t_hat: float
    per_locus: list[HKALocusResult]


@lru_cache(maxsize=64)
def _watterson_a(n: int) -> float:
    """Harmonic number a_n = sum(1/j for j = 1 .. n-1)."""
    return sum(1.0 / j for j in range(1, n))


@lru_cache(maxsize=64)
def _watterson_a2(n: int) -> float:
    """a2_n = sum(1/j^2 for j = 1 .. n-1)."""
    return sum(1.0 / (j * j) for j in range(1, n))


def hka_test(
    inputs: Sequence[HKALocusInput], mode: str = "pwd"
) -> HKATestResult:
    """Run the classic HKA test.

    Args:
        inputs: Per-locus data (at least 2 loci required).
        mode: ``"seg"`` for segregating-sites or ``"pwd"`` for pairwise
            differences.

    Returns:
        ``HKATestResult`` with chi-squared statistic, df, p-value, and
        per-locus breakdown including direction of deviation.

    Raises:
        ValueError: if fewer than 2 loci or total polymorphism is zero.
    """
    if len(inputs) < 2:
        raise ValueError("HKA test requires at least 2 loci")
    if mode not in ("seg", "pwd"):
        raise ValueError(f"mode must be 'seg' or 'pwd', got {mode!r}")

    total_poly = sum(inp.poly for inp in inputs)
    total_div = sum(inp.div for inp in inputs)

    if total_poly <= 0:
        raise ValueError("total polymorphism is zero; cannot estimate parameters")

    if mode == "seg":
        # Pool theta estimates via Watterson: sum(S_i / a(n_i)) estimates sum(theta_i)
        theta_sum = sum(inp.poly / _watterson_a(inp.n_seqs) for inp in inputs)
        t_plus_1 = total_div / theta_sum if theta_sum > 0 else total_div / total_poly
    else:
        t_plus_1 = total_div / total_poly

    t_hat = t_plus_1 - 1.0

    locus_results: list[HKALocusResult] = []
    total_chi2 = 0.0

    for inp in inputs:
        n = inp.n_seqs
        a_n = _watterson_a(n)
        a2_n = _watterson_a2(n)
        poly_scale = a_n if mode == "seg" else 1.0

        theta_i = (inp.poly + inp.div) / (poly_scale + t_plus_1)

        exp_poly = theta_i * poly_scale
        exp_div = theta_i * t_plus_1

        if mode == "seg":
            var_poly = theta_i * a_n + theta_i ** 2 * a2_n
        else:
            b1 = (n + 1) / (3.0 * (n - 1))
            b2 = 2.0 * (n * n + n + 3) / (9.0 * n * (n - 1))
            var_poly = theta_i * b1 + theta_i ** 2 * b2

        var_div = theta_i * t_plus_1 + theta_i ** 2

        chi2_poly = (inp.poly - exp_poly) ** 2 / var_poly if var_poly > 0 else 0.0
        chi2_div = (inp.div - exp_div) ** 2 / var_div if var_div > 0 else 0.0
        total_chi2 += chi2_poly + chi2_div

        direction = "excess_poly" if inp.poly > exp_poly else "deficit_poly"

        locus_results.append(
            HKALocusResult(
                locus=inp.locus,
                theta_hat=theta_i,
                obs_poly=inp.poly,
                exp_poly=exp_poly,
                var_poly=var_poly,
                obs_div=inp.div,
                exp_div=exp_div,
                var_div=var_div,
                chi2_poly=chi2_poly,
                chi2_div=chi2_div,
                direction=direction,
            )
        )

    df = len(inputs) - 1
    p_value = 1.0 - chi2.cdf(total_chi2, df)

    return HKATestResult(
        mode=mode,
        chi2=total_chi2,
        df=df,
        p_value=p_value,
        t_hat=t_hat,
        per_locus=locus_results,
    )
