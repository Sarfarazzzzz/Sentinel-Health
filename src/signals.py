"""Safety-signal detection: temporal change detection + descriptive disproportionality.

v4 - current design, arrived at by inspecting real output three times.

WHY IT LOOKS LIKE THIS.
v1 compared each device against ALL other devices. The top signal was a heating
pad with electrical failures, which is what a heating pad IS, not a discovery.
v2 stratified by medical specialty panel, which helped, but the top signals were
still tautologies (root canal resin over-reports sealing failures) and PRR values
reached 80,000: panels are dominated by one device type - the Dental panel is
~95% implant osseointegration reports - so every other failure mode has a
near-zero baseline and any device specialising in it divides by ~nothing.

The underlying problem is that cross-sectional disproportionality assumes devices
within a comparison group are exchangeable. Medical devices are not: each type has
characteristic failure modes by design, so "which device over-reports mode Y"
largely recovers "which device is built to do Y". No threshold fixes that.

v3 made the primary alert TEMPORAL: is this device reporting this failure mode
more than IT USED TO? Each device is its own control, so device-type confounding
cancels by construction and what surfaces is change rather than identity.

v4 adds two corrections found by investigating v3's own alerts:
- recall_field_action is excluded from alerting. Recall-remediation reports are
  administrative follow-ups to problems the FDA has ALREADY acted on; a spike
  means a recall campaign is running, which is the opposite of an early warning.
  They stay in the data as context.
- MIN_MONTH_REPORTS raised to 25. Alerts with n=6 produced z-scores above 20 that
  were measuring noise in a tiny denominator.

Reporting artifacts (new narrative templates, manufacturer filing changes, batch
retrospective filing) are detected separately in artifact_guard.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import device_meta

GOLD_DIR = Path("data/gold")

# --- temporal alerting (primary) ---
TREND_Z_MIN = 3.0         # z-score of current share vs the device's own baseline
MIN_MONTH_REPORTS = 25    # small denominators produce meaningless z-scores
BASELINE_MONTHS = 12
MIN_BASELINE_MONTHS = 6
RECENT_MONTHS = 6         # an alert is "active" if it fired in this trailing window
# Report TYPES that describe administrative follow-up rather than device failure.
NON_ALERTING_MODES = ["recall_field_action"]

# --- disproportionality (descriptive context) ---
MIN_REPORTS = 10
MIN_EXPECTED = 5.0        # excludes degenerate near-zero baselines (E << 1)
PRR_MIN = 2.0
EB05_MIN = 2.0
MIN_PANEL_SIZE = 500


# ---------------------------------------------------------------- context
def two_by_two(df: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Contingency counts within one panel (the peer group)."""
    g = df[df["panel"] == panel]
    ct = g.groupby(["product_code", "failure_mode"]).size().rename("a").reset_index()
    device_totals = g.groupby("product_code").size().rename("device_total")
    mode_totals = g.groupby("failure_mode").size().rename("mode_total")
    N = len(g)
    ct = ct.join(device_totals, on="product_code").join(mode_totals, on="failure_mode")
    ct["b"] = ct["device_total"] - ct["a"]
    ct["c"] = ct["mode_total"] - ct["a"]
    ct["d"] = N - ct["a"] - ct["b"] - ct["c"]
    ct["E"] = ct["device_total"] * ct["mode_total"] / N     # expected count
    ct["panel"] = panel
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


def add_eb_shrinkage(ct: pd.DataFrame) -> pd.DataFrame:
    """Gamma-Poisson EB with a method-of-moments prior fitted within panel."""
    a, E = ct["a"].to_numpy(float), ct["E"].to_numpy(float)
    rr = a / np.maximum(E, 1e-9)
    mean, var = rr.mean(), max(rr.var(), 1e-6)
    alpha, beta = mean ** 2 / var, mean / var
    ct["EBRR"] = (alpha + ct["a"]) / (beta + ct["E"])
    ct["EB05"] = sps.gamma.ppf(0.05, alpha + ct["a"], scale=1.0 / (beta + ct["E"]))
    return ct


def flag_disproportionate(ct: pd.DataFrame) -> pd.DataFrame:
    """Descriptive flag for characteristic associations - NOT the alert."""
    ct["disproportionate"] = ((ct["a"] >= MIN_REPORTS)
                              & (ct["E"] >= MIN_EXPECTED)
                              & (ct["PRR"] >= PRR_MIN)
                              & (ct["EB05"] >= EB05_MIN))
    return ct


# ---------------------------------------------------------------- alerting
def emerging_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Self-controlled change detection: each device is its own baseline."""
    # Recall-remediation reports are administrative follow-ups to problems the
    # FDA has already acted on - a spike means a recall campaign is underway,
    # not an emerging safety issue. Excluded from alerting, kept in context.
    df = df[~df["failure_mode"].isin(NON_ALERTING_MODES)]

    monthly = (df.groupby(["product_code", "failure_mode", "month"]).size()
                 .rename("n").reset_index())
    device_monthly = (df.groupby(["product_code", "month"]).size()
                        .rename("device_n").reset_index())
    m = monthly.merge(device_monthly, on=["product_code", "month"], how="left")
    # Share rather than raw count, so a device's overall reporting volume drifting
    # over time does not by itself create or mask an alert.
    m["share"] = m["n"] / m["device_n"].replace(0, np.nan)
    m = m.sort_values("month")

    def _z(g):
        base_mean = g["share"].rolling(BASELINE_MONTHS,
                                       min_periods=MIN_BASELINE_MONTHS).mean().shift(1)
        base_std = g["share"].rolling(BASELINE_MONTHS,
                                      min_periods=MIN_BASELINE_MONTHS).std().shift(1)
        g["baseline_share"] = base_mean
        g["trend_z"] = (g["share"] - base_mean) / base_std.replace(0, np.nan)
        return g

    m = m.groupby(["product_code", "failure_mode"], group_keys=False)[m.columns].apply(_z)
    m["alert"] = (m["trend_z"] >= TREND_Z_MIN) & (m["n"] >= MIN_MONTH_REPORTS)
    return m


def run(as_of: str | None = None) -> pd.DataFrame:
    """Compute signals using data up to `as_of` (YYYY-MM). None = everything.

    The as_of parameter keeps retrospective validation honest: signals are
    recomputed as they would have looked at that time, with no future data.
    """
    df = pd.read_parquet(GOLD_DIR / "events_extracted.parquet")
    df = df[~df["failure_mode"].isin(["unknown", "no_narrative"])]
    if as_of:
        df = df[df["month"] <= as_of]
    df = device_meta.annotate(df)

    # ---- descriptive context table ----
    panel_sizes = df["panel"].value_counts()
    panels = panel_sizes[panel_sizes >= MIN_PANEL_SIZE].index
    tables = [flag_disproportionate(
                  add_eb_shrinkage(add_disproportionality(two_by_two(df, p))))
              for p in panels]
    tables = [t for t in tables if not t.empty]
    ct = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    # ---- temporal alerts ----
    trends = device_meta.annotate(emerging_signals(df))

    if not ct.empty:
        names = device_meta.load()[["product_code", "device_name"]]
        ct = ct.merge(names, on="product_code", how="left")
        ct["device_name"] = ct["device_name"].fillna(ct["product_code"])

    if as_of is None:
        latest = trends["month"].max()
        recent_cut = (pd.Period(latest, freq="M") - RECENT_MONTHS).strftime("%Y-%m")
        active = trends[trends["alert"] & (trends["month"] >= recent_cut)]

        ct.to_parquet(GOLD_DIR / "signals.parquet", index=False)
        trends.to_parquet(GOLD_DIR / "trends.parquet", index=False)
        active.to_parquet(GOLD_DIR / "active_alerts.parquet", index=False)

        print("\nEMERGING SIGNALS (primary alert, self-controlled)")
        print(f"  {len(active):,} active alerts in the last {RECENT_MONTHS} months "
              f"of {int(trends['alert'].sum()):,} historical alerts")
        cols = ["month", "device_name", "failure_mode", "n", "share",
                "baseline_share", "trend_z"]
        if not active.empty:
            print(active.sort_values("trend_z", ascending=False)[cols]
                  .head(10).to_string(index=False))

        n_disp = int(ct["disproportionate"].sum()) if not ct.empty else 0
        print("\nDISPROPORTIONALITY (descriptive context)")
        print(f"  {n_disp:,} of {len(ct):,} pairs flagged "
              f"({n_disp/max(len(ct),1):.1%}) across {len(panels)} panels")
    return ct


if __name__ == "__main__":
    run()