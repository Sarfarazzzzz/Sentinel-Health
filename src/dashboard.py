"""Streamlit dashboard for Sentinel-Health. Run: streamlit run src/dashboard.py"""

from pathlib import Path

import pandas as pd
import streamlit as st

GOLD = Path("data/gold")

st.set_page_config(page_title="Sentinel — Device Safety Signals", layout="wide")
st.title("Sentinel: FDA Device Adverse-Event Early Warning")

signals = pd.read_parquet(GOLD / "signals.parquet")
trends = pd.read_parquet(GOLD / "trends.parquet")

flagged = signals[signals["signal"]].sort_values("EB05", ascending=False)
c1, c2, c3 = st.columns(3)
c1.metric("Device-problem pairs analyzed", f"{len(signals):,}")
c2.metric("Active signals", f"{len(flagged):,}")
c3.metric("Highest EB05", f"{flagged['EB05'].max():.1f}" if len(flagged) else "—")

st.subheader("Active safety signals (by shrunk lower bound, EB05)")
st.dataframe(
    flagged[["product_code", "failure_mode", "a", "PRR", "ROR", "chi2", "EBRR", "EB05"]]
    .rename(columns={"a": "reports"}).round(2),
    use_container_width=True, height=380,
)

st.subheader("Monthly trend for a device-problem pair")
pair = st.selectbox(
    "Pick a flagged pair",
    flagged.apply(lambda r: f"{r['product_code']} | {r['failure_mode']}", axis=1).tolist() or ["—"],
)
if pair != "—":
    code, mode = [s.strip() for s in pair.split("|")]
    t = trends[(trends["product_code"] == code) & (trends["failure_mode"] == mode)]
    st.line_chart(t.set_index("month")[["n"]], height=260)
    st.caption("Report counts per month. Baseline z-scores are in data/gold/trends.parquet.")

val_path = GOLD / "validation.parquet"
if val_path.exists():
    st.subheader("Retrospective recall validation")
    val = pd.read_parquet(val_path)
    hits = val["first_signal_month"].notna()
    a, b = st.columns(2)
    a.metric("Recalls flagged in advance", f"{hits.sum()} / {len(val)}")
    if hits.any():
        b.metric("Median lead time", f"{val.loc[hits, 'lead_time_months'].median():.0f} months")
    st.dataframe(val[["product_code", "recalling_firm", "recall_month",
                      "first_signal_month", "lead_time_months", "reason"]],
                 use_container_width=True)
