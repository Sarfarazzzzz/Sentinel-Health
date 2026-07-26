"""Smoke test: run dedup -> extract -> signals on synthetic data, no network.

    python -m tests.test_smoke
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src import dedup, extract, signals

random.seed(7)
np.random.seed(7)

SILVER = Path("data/silver")
GOLD = Path("data/gold")


def make_synthetic(n=3000) -> pd.DataFrame:
    devices = ["FRN", "DQY", "LZG", "MND", "OYC"]
    mfr_variants = {
        "acme medical": ["ACME MEDICAL INC", "Acme Medical, Inc.", "ACME MEDICAL"],
        "borealis devices": ["BOREALIS DEVICES LLC", "Borealis Devices"],
    }
    phrases = {
        "battery_power": "device battery depleted and unit powered off during therapy",
        "software": "software error code 41 displayed and the pump froze",
        "mechanical_break": "the catheter tip fractured during removal",
        "leak_seal": "fluid leak observed at the seal of the reservoir",
        "none": "patient reported discomfort, device functioned as intended",
    }
    rows = []
    months = pd.period_range("2024-01", "2025-12", freq="M").astype(str)
    for i in range(n):
        dev = random.choice(devices)
        fam = random.choice(list(mfr_variants))
        # Plant a real signal: device FRN develops a battery problem in 2025.
        month = random.choice(months)
        if dev == "FRN" and month >= "2024-09" and random.random() < 0.8:
            mode = "battery_power"
        else:
            mode = random.choice(list(phrases))
        filler_vocab = ["patient", "nurse", "reported", "hospital", "morning",
                        "evening", "home", "clinic", "replaced", "returned",
                        "evaluated", "technician", "follow", "up", "observed",
                        "unit", "serial", "lot", "therapy", "session", "during",
                        "after", "before", "immediately", "later", "physician"]
        filler = " ".join(random.sample(filler_vocab, 12))
        narrative = f"{phrases[mode]}. additional context: {filler} ref {random.randint(1, 99999)}"
        rows.append({
            "report_key": str(i),
            "date_received": month.replace("-", "") + "15",
            "date_of_event": None,
            "event_type": "Malfunction",
            "manufacturer_raw": random.choice(mfr_variants[fam]),
            "brand_name": "X", "generic_name": "pump",
            "product_code": dev,
            "product_problems": "",
            "narrative": narrative,
            "narrative_hash": str(hash(narrative)),
            "month": month,
        })
    # Add exact duplicates to exercise dedup.
    df = pd.DataFrame(rows)
    dupes = df.sample(200, random_state=7)
    return pd.concat([df, dupes], ignore_index=True)


def main():
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    make_synthetic().to_parquet(SILVER / "events.parquet", index=False)

    d = dedup.run()
    assert d["manufacturer"].nunique() <= 2, "manufacturer clustering failed"

    e = extract.run(mode="rules")
    assert (e["failure_mode"] == "battery_power").any(), "extraction failed"

    ct = signals.run()
    planted = ct[(ct["product_code"] == "FRN") & (ct["failure_mode"] == "battery_power")]
    assert not planted.empty and bool(planted["signal"].iloc[0]), \
        "planted FRN battery signal was not flagged"
    print("\nSMOKE TEST PASSED: planted signal detected, dedup and extraction ran.")


if __name__ == "__main__":
    main()
