"""Safety-signal detection with disproportionality statistics.

For every (product_code, failure_mode) pair we build the standard 2x2
pharmacovigilance table over a time window:

                    this failure mode   all other modes
  this device              a                  b
  all other devices        c                  d

and compute:
- PRR  (Proportional Reporting Ratio)     = (a/(a+b)) / (c/(c+d))
- ROR  (Reporting Odds Ratio)             = (a*d) / (b*c)
- chi2 (Yates-corrected)                  via scipy
- EBRR (empirical-Bayes shrunk relative reporting rate): a ~ Poisson(l*E),
  Gamma(alpha, beta) prior on l fitted by method of moments across all
  pairs, posterior mean (alpha+a)/(beta+E). EB05 = 5th posterior percentile,
  the conservative bound real signal-detection systems alert on.

A pair is flagged by the classic Evans criteria (PRR>=2, chi2>=4, a>=3)
AND EB05 >= 1.5. Also produces a monthly emerging-trend score per pair
(z-score of this month's count vs the trailing 12-month baseline).

Simplification vs. production systems: this is a Gamma-Poisson EB model with
a single Gamma prior, not DuMouchel's full MGPS mixture. Say exactly that in
interviews — knowing what you simplified is senior behavior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

GOLD_DIR = Path("data/gold")
MIN_REPORTS = 3


def two_by_two(df: pd.DataFrame) -> pd.DataFrame:
    ct = df.groupby(["product_code", "failure_mode"]).size().rename("a").reset_index()
    device_totals = df.groupby("product_code").size().rename("device_total")
    mode_totals = df.groupby("failure_mode").size().rename("mode_total")
    N = len(df)
    ct = ct.join(device_totals, on="product_code").join(mode_totals, on="failure_mode")
    ct["b"] = ct["device_total"] - ct["a"]
    ct["c"] = ct["mode_total"] - ct["a"]
    ct["d"] = N - ct["a"] - ct["b"] - ct["c"]
    ct["E"] = ct["device_total"] * ct["mode_total"] / N   # expected count
    return ct


def add_disproportionality(ct: pd.DataFrame) -> pd.DataFrame:
    a, b, c, d = (ct[k].astype(float) for k in "abcd")
    ct["PRR"] = (a / (a + b)) / ((c + 0.5) / (c + d + 0.5))
    ct["ROR"] = (a * d + 0.5) / (b * c + 0.5)
    ct["chi2"] = [
        sps.chi2_contingency([[ai, bi], [ci, di]], correction=True)[0]
        if min(ai + bi, ci + di) > 0 else 0.0
        for ai, bi, ci, di in zip(a, b, c, d)
    ]
    return ct


def fit_gamma_prior(a: np.ndarray, E: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Gamma prior on the relative reporting rate a/E."""
    rr = a / np.maximum(E, 1e-9)
    mean, var = rr.mean(), rr.var()
    var = max(var, 1e-6)
    alpha = mean ** 2 / var
    beta = mean / var
    return float(alpha), float(beta)


def add_eb_shrinkage(ct: pd.DataFrame) -> pd.DataFrame:
    alpha, beta = fit_gamma_prior(ct["a"].to_numpy(float), ct["E"].to_numpy(float))
    post_alpha = alpha + ct["a"]
    post_beta = beta + ct["E"]
    ct["EBRR"] = post_alpha / post_beta
    ct["EB05"] = sps.gamma.ppf(0.05, post_alpha, scale=1.0 / post_beta)
    ct.attrs["gamma_prior"] = (alpha, beta)
    return ct


def flag_signals(ct: pd.DataFrame) -> pd.DataFrame:
    ct["signal"] = (
        (ct["a"] >= MIN_REPORTS) & (ct["PRR"] >= 2.0)
        & (ct["chi2"] >= 4.0) & (ct["EB05"] >= 1.5)
    )
    return ct


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing-baseline z-score of monthly counts per (device, mode)."""
    m = (df.groupby(["product_code", "failure_mode", "month"]).size()
           .rename("n").reset_index().sort_values("month"))
    def _z(g):
        base_mean = g["n"].rolling(12, min_periods=6).mean().shift(1)
        base_std = g["n"].rolling(12, min_periods=6).std().shift(1)
        g["trend_z"] = (g["n"] - base_mean) / base_std.replace(0, np.nan)
        return g
    return m.groupby(["product_code", "failure_mode"], group_keys=False)[m.columns].apply(_z)


def run(as_of: str | None = None) -> pd.DataFrame:
    """Compute signals using data up to `as_of` (YYYY-MM). None = everything.

    The as_of parameter is what makes retrospective validation honest: the
    validator recomputes signals as they would have looked at each point in
    time, with no future data leaking in.
    """
    df = pd.read_parquet(GOLD_DIR / "events_extracted.parquet")
    df = df[df["failure_mode"] != "unknown"]
    if as_of:
        df = df[df["month"] <= as_of]
    ct = flag_signals(add_eb_shrinkage(add_disproportionality(two_by_two(df))))
    trends = monthly_trend(df)
    if as_of is None:
        ct.to_parquet(GOLD_DIR / "signals.parquet", index=False)
        trends.to_parquet(GOLD_DIR / "trends.parquet", index=False)
        top = ct[ct["signal"]].sort_values("EB05", ascending=False)
        print(f"{ct['signal'].sum()} flagged signals of {len(ct)} pairs. Top 10 by EB05:")
        print(top[["product_code", "failure_mode", "a", "PRR", "EB05"]].head(10).to_string(index=False))
    return ct


if __name__ == "__main__":
    run()
