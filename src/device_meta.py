"""Fetch FDA device classification metadata: product_code -> device name + panel.

The openFDA device classification endpoint maps each 3-letter product code to
a human-readable device name and a medical specialty panel (e.g. "Cardiovascular",
"Dental", "General Hospital"). Two uses:

1. Display: show "Heating Pad" instead of "IRT" everywhere.
2. Statistics: the panel is the PEER GROUP for disproportionality analysis.
   Comparing a heating pad against all devices (including dental implants and
   glucose sensors) makes any powered device look wildly disproportionate for
   electrical failures - that is confounding by device type, not a signal.
   Comparing it against other General Hospital devices is the real question.

The whole classification table is ~7k rows, so this fetches once and caches.

    python -m src.device_meta          # fetch/refresh the cache
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

CLASSIFICATION_URL = "https://api.fda.gov/device/classification.json"
PAGE = 1000
META_DIR = Path("data/meta")
META_PATH = META_DIR / "device_classification.parquet"
UNKNOWN_PANEL = "(unclassified)"


def _get(params: dict, max_retries: int = 5) -> dict:
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(CLASSIFICATION_URL, params=params, timeout=90)
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt * 3, 60))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return {"results": []}
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt * 5, 120))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"classification API failed: {last_err}")


def fetch() -> pd.DataFrame:
    """Download the full device classification table and cache it."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    rows, skip = [], 0
    while True:
        data = _get({"limit": PAGE, "skip": skip})
        results = data.get("results", [])
        rows.extend(results)
        print(f"  fetched {len(rows):,} classification records")
        if len(results) < PAGE:
            break
        skip += PAGE
        time.sleep(0.25)

    df = pd.DataFrame([{
        "product_code": r.get("product_code", ""),
        "device_name": r.get("device_name", ""),
        "panel": r.get("medical_specialty_description") or UNKNOWN_PANEL,
        "device_class": r.get("device_class", ""),
        "regulation_number": r.get("regulation_number", ""),
    } for r in rows])
    df = df[df["product_code"].str.len() > 0].drop_duplicates("product_code")
    df.to_parquet(META_PATH, index=False)
    print(f"Cached {len(df):,} product codes -> {META_PATH}")
    print(f"Panels: {df['panel'].nunique()}")
    return df


def load() -> pd.DataFrame:
    """Load the cached table, fetching it first if absent."""
    if not META_PATH.exists():
        print("Device classification cache missing - fetching...")
        return fetch()
    return pd.read_parquet(META_PATH)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add device_name and panel columns to any frame with a product_code."""
    meta = load()[["product_code", "device_name", "panel"]]
    out = df.merge(meta, on="product_code", how="left")
    out["device_name"] = out["device_name"].fillna("")
    out["panel"] = out["panel"].fillna(UNKNOWN_PANEL)
    # Product codes absent from the classification table keep their code as a
    # display name so nothing silently disappears from the dashboard.
    missing = out["device_name"] == ""
    out.loc[missing, "device_name"] = out.loc[missing, "product_code"]
    return out


if __name__ == "__main__":
    fetch()
