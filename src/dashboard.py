"""Streamlit dashboard for Sentinel-Health. Run: streamlit run src/dashboard.py

v5 - combined view. Retrospective validation showed the two detectors answer
different questions and neither is sufficient alone:

  temporal (self-controlled z-score)  -> EMERGING problems; caught 1/40 recalls
  cross-sectional (disproportionality)-> CHRONIC problems; 42% of recalled
                                         devices vs a 20% base rate (2.1x lift)

Recalls typically follow chronic accumulation, which is why the temporal
detector - designed to find change - missed them. The primary view now unions
both detectors and ranks by how many fired, rather than presenting one as the
alert and the other as context.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

GOLD = Path("data/gold")

st.set_page_config(page_title="Sentinel — Device Safety Signals", layout="wide")
st.title("Sentinel: FDA Device Adverse-Event Early Warning")
st.caption(
    "Two complementary detectors over FDA MAUDE adverse-event reports. "
    "**Temporal**: each device against its own 12-month baseline — finds *emerging* "
    "problems. **Cross-sectional**: each device against its specialty peers — finds "
    "*chronic* ones. MAUDE is passive surveillance: report volume reflects reporting "
    "behaviour as well as risk, so a signal is a prompt to investigate, never "
    "evidence that a device is unsafe."
)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(GOLD / name)


alerts = load("active_alerts.parquet")
trends = load("trends.parquet")
signals = load("signals.parquet")
has_guard = "artifact_risk" in alerts.columns


@st.cache_data
def build_combined() -> pd.DataFrame:
    """Union both detectors on (product_code, failure_mode)."""
    # Temporal: most recent alert per device-mode
    t = (alerts.sort_values("month")
               .drop_duplicates(["product_code", "failure_mode"], keep="last"))
    tcols = ["product_code", "failure_mode", "device_name", "panel", "month",
             "n", "share", "baseline_share", "trend_z"]
    if has_guard:
        tcols += ["artifact_risk", "artifact_reasons"]
    t = t[[c for c in tcols if c in t.columns]].copy()
    t["temporal_flag"] = True

    # Cross-sectional: disproportionate device-modes
    flag_col = "disproportionate" if "disproportionate" in signals.columns else "signal"
    c = signals[signals[flag_col]][
        ["product_code", "failure_mode", "device_name", "panel", "a", "E",
         "PRR", "EB05"]].copy()
    c["crosssectional_flag"] = True

    m = t.merge(c, on=["product_code", "failure_mode"], how="outer",
                suffixes=("", "_c"))
    for col in ("device_name", "panel"):
        if f"{col}_c" in m.columns:
            m[col] = m[col].fillna(m[f"{col}_c"])
            m = m.drop(columns=[f"{col}_c"])
    m["temporal_flag"] = m["temporal_flag"].fillna(False)
    m["crosssectional_flag"] = m["crosssectional_flag"].fillna(False)
    m["detectors"] = m["temporal_flag"].astype(int) + m["crosssectional_flag"].astype(int)
    if has_guard:
        m["artifact_risk"] = m["artifact_risk"].fillna("n/a")
        m["artifact_reasons"] = m["artifact_reasons"].fillna("")
    return m


combined = build_combined()

tab0, tab1, tab2 = st.tabs(
    ["Devices to investigate", "Emerging (temporal)", "Chronic (cross-sectional)"])

# ------------------------------------------------------------------ combined
with tab0:
    both = combined[combined["detectors"] == 2]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Device-problem pairs flagged", f"{len(combined):,}")
    c2.metric("Flagged by both detectors", f"{len(both):,}")
    c3.metric("Emerging only", f"{int((combined['detectors'].eq(1) & combined['temporal_flag']).sum()):,}")
    c4.metric("Chronic only", f"{int((combined['detectors'].eq(1) & combined['crosssectional_flag']).sum()):,}")

    st.subheader("Ranked by how many detectors fired")
    colf1, colf2 = st.columns(2)
    with colf1:
        which = st.radio("Show", ["Both detectors", "Any detector",
                                  "Emerging only", "Chronic only"],
                         horizontal=True)
    with colf2:
        hide_artifacts = st.checkbox(
            "Hide known reporting-artifact patterns", value=True) if has_guard else False

    view = combined
    if which == "Both detectors":
        view = view[view["detectors"] == 2]
    elif which == "Emerging only":
        view = view[view["detectors"].eq(1) & view["temporal_flag"]]
    elif which == "Chronic only":
        view = view[view["detectors"].eq(1) & view["crosssectional_flag"]]
    if hide_artifacts and has_guard:
        view = view[view["artifact_risk"].isin(["low", "n/a"])]

    show_cols = [c for c in ["device_name", "failure_mode", "panel", "detectors",
                             "temporal_flag", "trend_z", "month",
                             "crosssectional_flag", "PRR", "EB05", "a",
                             "artifact_risk"] if c in view.columns]
    st.dataframe(
        view.sort_values(["detectors", "trend_z"], ascending=[False, False])[show_cols]
            .rename(columns={"device_name": "Device", "failure_mode": "Failure mode",
                             "panel": "Panel", "detectors": "# detectors",
                             "temporal_flag": "Emerging", "trend_z": "z",
                             "month": "Alert month",
                             "crosssectional_flag": "Chronic", "a": "Reports",
                             "artifact_risk": "Artifact risk"}).round(2),
        use_container_width=True, height=420)

    with st.expander("What the two detectors mean, and how they were validated"):
        st.markdown("""
**Emerging (temporal).** For each device and failure mode, what share of that
device's monthly reports the mode accounts for, compared against that device's
own trailing 12-month baseline as a z-score. Flags at z ≥ 3 with ≥ 25 reports.
Using each device as its own control removes device-type confounding — a heating
pad is *expected* to have electrical failures, so it never alerts, but one whose
electrical share doubles does.

**Chronic (cross-sectional).** Disproportionality within medical specialty
panels: is this device's rate of a failure mode unusual compared to its peers?
PRR with empirical-Bayes shrinkage (EB05 = conservative lower bound), plus a
minimum expected count so near-zero baselines cannot produce absurd ratios.

**Why both.** Retrospective validation against real FDA recalls, with no
lookahead: the temporal detector flagged **1 of 40** recalls in advance. That is
structural rather than a bug — recalls typically follow *chronic* problems
accumulating at a stable rate, and a device with a persistently high failure
share has a z-score near zero. Testing the cross-sectional detector instead:
recalled devices were flagged at **42% versus a 20% base rate (2.1× lift)**,
though only 12 recalls had six or more months of prior data, so this is
suggestive rather than conclusive.

**Artifact risk.** Text-derived surveillance detects changes in how events are
*reported* as well as changes in the events. Alerts are checked for new narrative
templates, manufacturer filing shifts, total-volume spikes, and coordinated
cross-device filing changes. Flagged alerts are annotated, not removed.
        """)

# ------------------------------------------------------------------ temporal
with tab1:
    st.subheader("Emerging: devices above their own baseline")
    a = alerts if not has_guard else (
        alerts[alerts["artifact_risk"] == "low"]
        if st.checkbox("Hide reporting-artifact patterns", value=True, key="t1")
        else alerts)
    tcols = [c for c in ["month", "device_name", "failure_mode", "panel", "n",
                         "share", "baseline_share", "trend_z", "artifact_risk",
                         "artifact_reasons"] if c in a.columns]
    st.dataframe(a.sort_values("trend_z", ascending=False)[tcols].round(3),
                 use_container_width=True, height=340)

    if not a.empty:
        labels = sorted({f"{r['device_name']} | {r['failure_mode']}"
                         for _, r in a.iterrows()})
        pick = st.selectbox("History for one alert", labels)
        dev, mode = [s.strip() for s in pick.split("|", 1)]
        row = a[(a["device_name"] == dev) & (a["failure_mode"] == mode)].iloc[0]
        t = trends[(trends["product_code"] == row["product_code"])
                   & (trends["failure_mode"] == mode)].sort_values("month")
        series = [c for c in ["share", "baseline_share"] if c in t.columns]
        if not t.empty and series:
            st.line_chart(t.set_index("month")[series], height=260)
            st.caption("Monthly share of this device's reports vs its own rolling baseline.")

# ------------------------------------------------------------ cross-sectional
with tab2:
    st.subheader("Chronic: devices unusual against their specialty peers")
    st.markdown(
        "This partly recovers what a device *is* — root canal resin over-reports "
        "sealing failures — which is why it was initially treated as context. "
        "Validation reversed that reading: it is the detector that flags recalled "
        "devices at 2.1× the base rate."
    )
    flag_col = "disproportionate" if "disproportionate" in signals.columns else "signal"
    disp = signals[signals[flag_col]].sort_values("EB05", ascending=False)
    panels = ["All panels"] + sorted(signals["panel"].dropna().unique().tolist())
    sel = st.selectbox("Medical specialty panel", panels)
    dv = disp if sel == "All panels" else disp[disp["panel"] == sel]
    dcols = [c for c in ["device_name", "failure_mode", "panel", "a", "E",
                         "PRR", "ROR", "EBRR", "EB05"] if c in dv.columns]
    st.dataframe(dv[dcols].rename(columns={
        "device_name": "Device", "failure_mode": "Failure mode",
        "panel": "Panel", "a": "Reports", "E": "Expected"}).round(2),
        use_container_width=True, height=380)

# ------------------------------------------------------------------ validation
val_path = GOLD / "validation.parquet"
xval_path = GOLD / "validation_crosssectional.parquet"
if val_path.exists() or xval_path.exists():
    st.subheader("Retrospective validation against FDA recalls")
    c1, c2 = st.columns(2)
    if val_path.exists():
        val = load("validation.parquet")
        hits = val["first_signal_month"].notna()
        c1.metric("Temporal: recalls flagged in advance",
                  f"{int(hits.sum())} / {len(val)}")
    if xval_path.exists():
        xval = load("validation_crosssectional.parquet")
        c2.metric("Cross-sectional: recalls flagged in advance",
                  f"{int(xval['flagged_before'].sum())} / {len(xval)}",
                  help="Base rate across all devices was 20%")
    st.caption(
        "Signals recomputed as of the month before each recall, so no future data "
        "influences the result. Small sample for the cross-sectional test (12 "
        "recalls had sufficient prior history), so treat as suggestive."
    )