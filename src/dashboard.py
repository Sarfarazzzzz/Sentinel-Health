"""Streamlit dashboard for Sentinel-Health. Run: streamlit run src/dashboard.py

v4 - emerging (temporal) signals lead, with reporting-artifact annotations.
Disproportionality is kept as descriptive context in a second tab.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

GOLD = Path("data/gold")

st.set_page_config(page_title="Sentinel — Device Safety Signals", layout="wide")
st.title("Sentinel: FDA Device Adverse-Event Early Warning")
st.caption(
    "Emerging-signal detection over FDA MAUDE adverse-event reports. Each device "
    "is compared against **its own 12-month baseline**, so what surfaces is change "
    "rather than a device's inherent characteristics. MAUDE is passive surveillance: "
    "report volume reflects reporting behaviour as well as risk, and a signal is a "
    "prompt to investigate, never evidence that a device is unsafe."
)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(GOLD / name)


alerts_all = load("active_alerts.parquet")
trends = load("trends.parquet")
signals = load("signals.parquet")

has_guard = "artifact_risk" in alerts_all.columns

tab1, tab2 = st.tabs(["Emerging signals", "Disproportionality (context)"])

with tab1:
    if has_guard:
        clean_n = int((alerts_all["artifact_risk"] == "low").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active alerts", f"{len(alerts_all):,}")
        c2.metric("No artifact pattern", f"{clean_n:,}")
        c3.metric("Devices involved", f"{alerts_all['product_code'].nunique():,}")
        c4.metric("Highest z-score",
                  f"{alerts_all['trend_z'].max():.1f}" if len(alerts_all) else "—")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Active alerts", f"{len(alerts_all):,}")
        c2.metric("Devices involved", f"{alerts_all['product_code'].nunique():,}")
        c3.metric("Highest z-score",
                  f"{alerts_all['trend_z'].max():.1f}" if len(alerts_all) else "—")

    st.subheader("Devices reporting a failure mode above their own baseline")

    alerts = alerts_all
    if has_guard:
        only_clean = st.checkbox(
            "Hide alerts with reporting-artifact patterns", value=True)
        if only_clean:
            alerts = alerts_all[alerts_all["artifact_risk"] == "low"]

    cols = ["month", "device_name", "failure_mode", "panel", "n", "share",
            "baseline_share", "trend_z"]
    if has_guard:
        cols += ["artifact_risk", "artifact_reasons"]
    cols = [c for c in cols if c in alerts.columns]

    show = alerts.sort_values("trend_z", ascending=False)[cols].rename(columns={
        "month": "Month", "device_name": "Device", "failure_mode": "Failure mode",
        "panel": "Panel", "n": "Reports", "share": "Share this month",
        "baseline_share": "Baseline share", "trend_z": "z",
        "artifact_risk": "Artifact risk", "artifact_reasons": "Why"})
    if show.empty:
        st.info("No alerts match the current filter.")
    else:
        st.dataframe(show.round(3), use_container_width=True, height=380)

    with st.expander("How to read this"):
        st.markdown("""
For each device and failure mode, the pipeline computes what share of that
device's monthly reports the failure mode accounts for, then compares it against
that device's own trailing 12-month baseline as a z-score.

- **Share this month** vs **Baseline share** — the comparison being made.
- **z** — how many standard deviations above its own history. z ≥ 3 flags.

Using each device as its own control removes device-type confounding: a heating
pad is expected to have electrical failures, so that never alerts — but a heating
pad whose electrical share *doubles* does.

**Artifact risk.** Text-derived surveillance detects changes in how events are
reported as well as changes in the events. Alerts are checked for three patterns:
a narrative template that never appeared in the baseline (a manufacturer changed
wording), one manufacturer suddenly supplying most reports (a filing-practice
change), and a spike in the device's total report volume (batch retrospective
filing of older events). Flagged alerts are annotated rather than removed — a
batch filing of real events is still worth seeing, it just is not emerging.
        """)

    if not alerts.empty:
        st.subheader("History for one alert")
        labels = sorted({f"{r['device_name']} | {r['failure_mode']}"
                         for _, r in alerts.iterrows()})
        pick = st.selectbox("Pick an alert", labels)
        dev, mode = [s.strip() for s in pick.split("|", 1)]
        row = alerts[(alerts["device_name"] == dev)
                     & (alerts["failure_mode"] == mode)].iloc[0]
        t = trends[(trends["product_code"] == row["product_code"])
                   & (trends["failure_mode"] == mode)].sort_values("month")
        series = [c for c in ["share", "baseline_share"] if c in t.columns]
        if not t.empty and series:
            st.line_chart(t.set_index("month")[series], height=260)
            st.caption("Monthly share of this device's reports, against its own "
                       "rolling baseline.")
        else:
            st.info("No monthly series available for this pair.")
        if has_guard and row.get("artifact_reasons", "none detected") != "none detected":
            st.warning(f"Artifact patterns detected: {row['artifact_reasons']}")

with tab2:
    st.subheader("Characteristic associations (descriptive, not alerts)")
    st.markdown(
        "Disproportionality analysis within medical specialty panels. This mostly "
        "recovers what a device *is* — root canal resin over-reports sealing "
        "failures, X-ray units over-report imaging failures — which is useful "
        "context but tautological as an early warning, which is why it does not "
        "drive alerting. A minimum expected count excludes degenerate near-zero "
        "baselines that produced PRR values in the tens of thousands."
    )
    flag_col = "disproportionate" if "disproportionate" in signals.columns else "signal"
    disp = signals[signals[flag_col]].sort_values("EB05", ascending=False)

    panels = ["All panels"] + sorted(signals["panel"].dropna().unique().tolist())
    sel = st.selectbox("Medical specialty panel", panels)
    view = disp if sel == "All panels" else disp[disp["panel"] == sel]

    dcols = [c for c in ["device_name", "failure_mode", "panel", "a", "E",
                         "PRR", "ROR", "EBRR", "EB05"] if c in view.columns]
    st.dataframe(
        view[dcols].rename(columns={
            "device_name": "Device", "failure_mode": "Failure mode",
            "panel": "Panel", "a": "Reports", "E": "Expected"}).round(2),
        use_container_width=True, height=380,
    )
    with st.expander("What these columns mean"):
        st.markdown("""
- **Reports** — reports for this device-and-failure-mode pair; **Expected** — how
  many its panel peers' rates would predict.
- **PRR** — proportional reporting ratio: observed over expected, within panel.
- **ROR** — the same comparison expressed as odds; higher than PRR by construction.
- **EBRR / EB05** — empirical-Bayes shrunk rate and its 5th percentile. EB05 is
  the conservative bound; small-count pairs are pulled toward the panel average.
        """)

val_path = GOLD / "validation.parquet"
if val_path.exists():
    st.subheader("Retrospective recall validation")
    val = load("validation.parquet")
    hits = val["first_signal_month"].notna()
    a, b = st.columns(2)
    a.metric("Recalls flagged in advance", f"{hits.sum()} / {len(val)}")
    if hits.any():
        b.metric("Median lead time",
                 f"{val.loc[hits, 'lead_time_months'].median():.0f} months")
    st.dataframe(val, use_container_width=True)