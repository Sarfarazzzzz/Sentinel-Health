"""Flag alerts that are likely reporting artifacts rather than device changes.

Text-derived surveillance detects changes in how events are REPORTED as well as
changes in the events themselves. Four patterns, all observed in real output:

1. NEW TEMPLATE - the dominant narrative opening in the alert month never
   appeared in the baseline. A manufacturer adopted new wording, reclassifying
   reports without any device changing. (Observed: Fresenius Kabi infusion pumps.)
2. MANUFACTURER SHIFT - one manufacturer supplies most of the alert month's
   reports but was minor or absent in the baseline. (Observed: Philips
   Respironics recall returns.)
3. VOLUME SPIKE - the device's TOTAL report volume spikes alongside the failure
   mode: the fingerprint of batch retrospective filing, where events are real but
   did not occur in the month they were filed. (Observed: Dexcom CGM, 54k reports
   filed in one month for a known coding issue.)
4. COORDINATED FILING CHANGE (cross-device) - one manufacturer shows new
   narrative templates across several product codes in the same month. Checks
   1 and 2 are per-device and miss this: an Olympus filing change surfaced as
   five independent endoscope "signals", each individually looking clean because
   Olympus was already the baseline leader for every scope type.

Alerts are annotated, not deleted: a batch filing of real events is still worth
seeing, it just is not an emerging problem. Adjudication stays with the human,
which is how real pharmacovigilance operates.

    python -m src.artifact_guard
"""

from pathlib import Path

import pandas as pd

GOLD_DIR = Path("data/gold")
PREFIX = 90
BASELINE_MONTHS = 6
MFR_SHIFT_MIN = 0.6       # one manufacturer holds this share in the alert month
MFR_BASELINE_MAX = 0.3    # ...but held at most this share in the baseline
VOLUME_SPIKE_RATIO = 2.0  # device total volume vs its own trailing median
CROSS_DEVICE_MIN = 3      # product codes with new templates from one manufacturer


def _month_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Dominant narrative template and manufacturer per (device, mode, month)."""
    d = df[["product_code", "failure_mode", "month", "narrative", "manufacturer"]].copy()
    d["prefix"] = d["narrative"].str.slice(0, PREFIX)

    pc = (d.groupby(["product_code", "failure_mode", "month", "prefix"])
            .size().rename("n").reset_index())
    tot = (pc.groupby(["product_code", "failure_mode", "month"])["n"]
             .sum().rename("mode_total").reset_index())
    top_prefix = (pc.sort_values("n", ascending=False)
                    .drop_duplicates(["product_code", "failure_mode", "month"])
                    .rename(columns={"prefix": "top_prefix", "n": "top_prefix_n"}))
    top_prefix = top_prefix.merge(tot, on=["product_code", "failure_mode", "month"])

    mc = (d.groupby(["product_code", "failure_mode", "month", "manufacturer"])
            .size().rename("n").reset_index())
    top_mfr = (mc.sort_values("n", ascending=False)
                 .drop_duplicates(["product_code", "failure_mode", "month"])
                 .rename(columns={"manufacturer": "top_mfr", "n": "top_mfr_n"}))
    top_mfr = top_mfr.merge(tot, on=["product_code", "failure_mode", "month"],
                            suffixes=("", "_t"))
    top_mfr["top_mfr_share"] = top_mfr["top_mfr_n"] / top_mfr["mode_total"]

    return top_prefix.merge(
        top_mfr[["product_code", "failure_mode", "month", "top_mfr", "top_mfr_share"]],
        on=["product_code", "failure_mode", "month"], how="left")


def _device_volume(df: pd.DataFrame) -> pd.DataFrame:
    v = df.groupby(["product_code", "month"]).size().rename("device_total").reset_index()
    v = v.sort_values("month")
    v["median_prior"] = (v.groupby("product_code")["device_total"]
                          .transform(lambda s: s.rolling(BASELINE_MONTHS,
                                                         min_periods=3).median().shift(1)))
    return v


def annotate(alerts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    prof = _month_profiles(events)
    vol = _device_volume(events)
    prof_idx = prof.set_index(["product_code", "failure_mode", "month"])

    rows = []
    for _, a in alerts.iterrows():
        key = (a["product_code"], a["failure_mode"], a["month"])
        reasons, risk = [], "low"
        new_template, top_mfr = False, None

        if key in prof_idx.index:
            p = prof_idx.loc[key]
            top_mfr = p["top_mfr"]
            months = sorted(prof[(prof["product_code"] == a["product_code"])
                                 & (prof["failure_mode"] == a["failure_mode"])
                                 & (prof["month"] < a["month"])]["month"])[-BASELINE_MONTHS:]
            base = events[(events["product_code"] == a["product_code"])
                          & (events["failure_mode"] == a["failure_mode"])
                          & (events["month"].isin(months))]
            seen = 0
            if not base.empty:
                seen = base["narrative"].str.slice(0, PREFIX).eq(p["top_prefix"]).sum()
            if seen == 0 and p["top_prefix_n"] >= 10:
                new_template = True
                reasons.append("new narrative template")
                risk = "high"

            base_prof = prof[(prof["product_code"] == a["product_code"])
                             & (prof["failure_mode"] == a["failure_mode"])
                             & (prof["month"].isin(months))]
            base_share = (base_prof.loc[base_prof["top_mfr"] == p["top_mfr"],
                                        "top_mfr_share"].mean())
            if (p["top_mfr_share"] >= MFR_SHIFT_MIN
                    and (pd.isna(base_share) or base_share <= MFR_BASELINE_MAX)):
                reasons.append(f"manufacturer shift ({p['top_mfr']})")
                risk = "high"

        vrow = vol[(vol["product_code"] == a["product_code"])
                   & (vol["month"] == a["month"])]
        if not vrow.empty and pd.notna(vrow["median_prior"].iloc[0]):
            ratio = vrow["device_total"].iloc[0] / max(vrow["median_prior"].iloc[0], 1)
            if ratio >= VOLUME_SPIKE_RATIO:
                reasons.append(f"total volume spike ({ratio:.1f}x)")
                risk = "high" if risk == "high" else "medium"

        rows.append({**a.to_dict(),
                     "artifact_risk": risk,
                     "artifact_reasons": "; ".join(reasons) or "none detected",
                     "_new_template": new_template,
                     "_top_mfr": top_mfr})

    out = pd.DataFrame(rows)
    return _cross_device_pass(out)


def _cross_device_pass(out: pd.DataFrame) -> pd.DataFrame:
    """Pattern 4: one manufacturer, new templates across several devices, one month.

    Per-device checks miss this entirely - if a manufacturer already dominates
    every product line it supplies, no 'shift' is detectable within any single
    device, yet the coordinated timing across product codes is the giveaway.
    """
    if out.empty or "_new_template" not in out.columns:
        return out.drop(columns=["_new_template", "_top_mfr"], errors="ignore")

    flagged = out[out["_new_template"] & out["_top_mfr"].notna()]
    if flagged.empty:
        return out.drop(columns=["_new_template", "_top_mfr"])

    counts = (flagged.groupby(["_top_mfr", "month"])["product_code"]
                     .nunique().rename("n_devices").reset_index())
    coordinated = counts[counts["n_devices"] >= CROSS_DEVICE_MIN]

    for _, c in coordinated.iterrows():
        mask = (out["_top_mfr"] == c["_top_mfr"]) & (out["month"] == c["month"])
        note = (f"coordinated manufacturer filing change "
                f"({c['_top_mfr']}, {int(c['n_devices'])} device types)")
        out.loc[mask, "artifact_reasons"] = out.loc[mask, "artifact_reasons"].apply(
            lambda r: note if r == "none detected" else f"{r}; {note}")
        out.loc[mask & (out["artifact_risk"] == "low"), "artifact_risk"] = "medium"
        print(f"  cross-device: {note} -> {int(mask.sum())} alerts annotated")

    return out.drop(columns=["_new_template", "_top_mfr"])


def run() -> pd.DataFrame:
    alerts = pd.read_parquet(GOLD_DIR / "active_alerts.parquet")
    events = pd.read_parquet(GOLD_DIR / "events_extracted.parquet")
    events = events[~events["failure_mode"].isin(["unknown", "no_narrative"])]
    out = annotate(alerts, events)
    out.to_parquet(GOLD_DIR / "active_alerts.parquet", index=False)
    print("\n" + out["artifact_risk"].value_counts().to_string())
    print(f"\n{(out['artifact_risk'] == 'low').sum()} alerts with no artifact "
          f"pattern detected -> these are the ones worth investigating first")
    return out


if __name__ == "__main__":
    run()
