"""""Deduplicate MAUDE reports and normalize manufacturer names.

v2 - performance fix for multi-million-row scale:
- The v1 fuzzy pass was a pure-Python pairwise loop with pandas label
  lookups per comparison: fine at smoke-test scale, hours at real scale.
  v2 batches each block through rapidfuzz.process.cdist (multithreaded
  C++), uses a positional numpy mask, and compares only the first
  COMPARE_CHARS characters of each narrative (multi-reporter duplicates
  are near-identical from the start; document this simplification).
- Also includes the acronym-safe name normalization ("I.T.S. GMBH" no
  longer collapses to empty) and an "(unknown)" bucket for blank names.

Thresholds are unchanged: tune them against your own inspected samples.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SILVER_DIR = Path("data/silver")
LEGAL_SUFFIXES = re.compile(
    r"\b(incorporated|inc|corp|corporation|company|co|ltd|llc|plc|gmbh|sa|ag|"
    r"limited|holdings?|group|usa|international|intl)\b\.?",
    re.IGNORECASE,
)
NAME_CLUSTER_THRESHOLD = 90     # token_sort_ratio to merge two manufacturer spellings
NARRATIVE_DUP_THRESHOLD = 95    # token_set_ratio to call two narratives the same event
COMPARE_CHARS = 500             # narrative prefix length used for fuzzy comparison
MAX_BLOCK = 200                 # skip pathologically large blocks (tune/inspect)


def normalize_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # collapse runs of single letters into acronyms: "i t s" -> "its"
    s = re.sub(r"\b((?:[a-z] )+[a-z])\b", lambda m: m.group(1).replace(" ", ""), s)
    s = LEGAL_SUFFIXES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "(unknown)"


def canonical_map(names: pd.Series) -> dict[str, str]:
    """Cluster normalized names; canonical = most frequent spelling in cluster."""
    counts = names.value_counts()
    uniques = list(counts.index)
    mapping: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    for n in uniques:
        blocks.setdefault(n.split(" ")[0] if n else "", []).append(n)
    for block in blocks.values():
        assigned: list[tuple[str, str]] = []
        for n in block:                        # frequency-ordered within block
            placed = False
            for member, canon in assigned:
                if fuzz.token_sort_ratio(n, member) >= NAME_CLUSTER_THRESHOLD:
                    mapping[n] = canon
                    placed = True
                    break
            if not placed:
                mapping[n] = n
                assigned.append((n, n))
    return mapping


MAX_HASH_COPIES = 5   # narratives appearing more often are templates, not dupes

def drop_duplicate_reports(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    # Pass 0: identical report_key = same report fetched twice (ingestion artifact)
    df = df.drop_duplicates(subset=["report_key"]).reset_index(drop=True)
    n_keys = len(df)

    # Identify duplicate-eligible rows: real text, appearing only a few times.
    # High-frequency identical narratives are manufacturer TEMPLATES describing
    # distinct events - never dedup those on text.
    hash_counts = df["narrative_hash"].map(df["narrative_hash"].value_counts())
    has_text = df["narrative"].str.len().fillna(0) > 40
    eligible = has_text & (hash_counts >= 2) & (hash_counts <= MAX_HASH_COPIES)

    # Pass 1: exact dedup on eligible rows only, with event date in the key
    deduped = df[eligible].drop_duplicates(
        subset=["manufacturer", "product_code", "narrative_hash", "date_of_event"])
    df = pd.concat([df[~eligible], deduped], ignore_index=True)
    n1 = len(df)
    print(f"  report_key dedup: {n0:,} -> {n_keys:,}")
    print(f"  template-aware exact dedup: {n_keys:,} -> {n1:,} "
          f"({eligible.sum():,} rows were duplicate-eligible)")

    # Pass 2: fuzzy pass, restricted to the same eligible population
    keep = np.ones(n1, dtype=bool)
    hash_counts = df["narrative_hash"].map(df["narrative_hash"].value_counts())
    fuzz_ok = ((df["narrative"].str.len().fillna(0) > 40)
               & (hash_counts <= MAX_HASH_COPIES)).to_numpy()
    prefix = df["narrative"].str.slice(0, COMPARE_CHARS).to_numpy()

    groups = df[fuzz_ok].groupby(
        ["manufacturer", "product_code", "month"], sort=False).indices
    n_groups = len(groups)
    done = 0
    for positions in groups.values():
        done += 1
        if done % 100_000 == 0:
            print(f"  fuzzy pass: {done:,}/{n_groups:,} blocks, "
                  f"removed so far: {(~keep).sum():,}")
        k = len(positions)
        if k < 2 or k > MAX_BLOCK:
            continue
        texts = prefix[positions]
        scores = process.cdist(texts, texts, scorer=fuzz.token_set_ratio,
                               score_cutoff=NARRATIVE_DUP_THRESHOLD, workers=-1)
        for i in range(k):
            if not keep[positions[i]]:
                continue
            for j in range(i + 1, k):
                if keep[positions[j]] and scores[i][j] >= NARRATIVE_DUP_THRESHOLD:
                    keep[positions[j]] = False

    df = df[keep]
    stats = {
        "input_rows": n0,
        "after_report_key_dedup": n_keys,
        "after_exact_dedup": n1,
        "after_fuzzy_dedup": len(df),
        "dedup_rate_pct": round(100 * (n0 - len(df)) / max(n0, 1), 2),
    }
    return df, stats


def run() -> pd.DataFrame:
    df = pd.read_parquet(SILVER_DIR / "events.parquet")
    print(f"loaded {len(df):,} rows")
    df["manufacturer_norm"] = df["manufacturer_raw"].map(normalize_name)
    print("normalized names; clustering...")
    mapping = canonical_map(df["manufacturer_norm"])
    df["manufacturer"] = df["manufacturer_norm"].map(mapping)
    print(f"clustered {df['manufacturer_norm'].nunique():,} normalized names "
          f"-> {df['manufacturer'].nunique():,} canonical")
    df, stats = drop_duplicate_reports(df)
    out = SILVER_DIR / "events_dedup.parquet"
    df.to_parquet(out, index=False)
    print(f"Dedup stats: {stats}")
    print(f"Manufacturer spellings collapsed: "
          f"{df['manufacturer_raw'].nunique():,} raw -> {df['manufacturer'].nunique():,} canonical")
    return df


if __name__ == "__main__":
    run()
