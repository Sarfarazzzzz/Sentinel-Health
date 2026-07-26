"""Score a LOCAL LLM (via Ollama) against your hand labels, next to rules.

v2 - hardened after the silent-failure incident:
- Verifies the requested model actually exists on the Ollama server before
  starting (a wrong model name previously 404'd on all 300 rows while the
  silent fallback made it look like the LLM answered 'unknown' every time).
- Errors print loudly on first occurrence per row.
- Early circuit breaker: aborts after 10 rows if outputs look like
  fallbacks, instead of burning a full run.

Setup:
    ollama serve            # keep running (or have the Ollama app open)
    ollama pull qwen2.5:3b

Usage (from project root):
    python eval_llm_ollama.py data/labels_todo.csv
    python eval_llm_ollama.py data/labels_todo.csv --model qwen2.5:7b
"""

import argparse
import json
import sys
import time

import pandas as pd
import requests

from src.extract import classify_rules, FAILURE_KEYWORDS

OLLAMA_BASE = "http://localhost:11434"

PROMPT = """You are extracting structured data from an FDA medical device adverse event narrative.

Narrative:
\"\"\"{narrative}\"\"\"

Respond with ONLY a JSON object, no other text, with keys:
- failure_mode: one of {modes} or "unknown"
- harm: one of ["death", "serious_injury", "injury", "malfunction_only"]
"""

FALLBACK = {"failure_mode": "unknown", "harm": "malfunction_only"}


def check_server_and_model(model: str) -> None:
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"ERROR: cannot reach Ollama at {OLLAMA_BASE} ({e}).\n"
                 "Is 'ollama serve' running (or the Ollama app open)?")
    available = [m["name"] for m in resp.json().get("models", [])]
    if model not in available:
        sys.exit(f"ERROR: model '{model}' is not on this Ollama server.\n"
                 f"Available: {available}\n"
                 f"Either run:  ollama pull {model}\n"
                 f"or rerun with:  --model {available[0] if available else '<name>'}")
    print(f"Ollama OK, model '{model}' found.")


def classify_ollama(narrative: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            narrative=narrative[:3000], modes=list(FAILURE_KEYWORDS))}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300)
            resp.raise_for_status()
            d = json.loads(resp.json()["message"]["content"])
            return {"failure_mode": str(d.get("failure_mode", "unknown")).strip(),
                    "harm": str(d.get("harm", "malfunction_only")).strip()}
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    print(f"  WARNING: LLM call failed after retries "
          f"({type(last_err).__name__}: {last_err})")
    return dict(FALLBACK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv", nargs="?", default="data/labels_todo.csv")
    ap.add_argument("--model", default="qwen2.5:3b")
    args = ap.parse_args()

    check_server_and_model(args.model)

    df = pd.read_csv(args.labels_csv, dtype={"report_key": str}).fillna("")
    df = df[df["failure_mode"].str.len() > 0].reset_index(drop=True)
    print(f"{len(df)} labeled rows | model: {args.model}")

    rules = pd.DataFrame([classify_rules(n, "") for n in df["narrative"]])

    llm_rows, t0 = [], time.time()
    for i, n in enumerate(df["narrative"], 1):
        llm_rows.append(classify_ollama(n, args.model))
        # Circuit breaker: if the first 10 rows are all fallback-shaped,
        # the endpoint is effectively dead - stop before wasting the run.
        if i == 10 and all(r == FALLBACK for r in llm_rows):
            sys.exit("\nERROR: first 10 LLM outputs are all fallback values - "
                     "calls are failing. See warnings above; fix and rerun.")
        if i % 25 == 0:
            rate = i / (time.time() - t0)
            eta = (len(df) - i) / max(rate, 0.01)
            print(f"  llm: {i}/{len(df)}  (~{eta/60:.0f} min remaining)")
    llm = pd.DataFrame(llm_rows)

    fallback_share = (llm["failure_mode"] == "unknown").mean()
    if fallback_share > 0.95:
        sys.exit("\nERROR: >95% of LLM outputs are 'unknown' - results look "
                 "like endpoint failure, not classification. Not writing results.")

    print("\n=== Accuracy ===")
    for col in ["failure_mode", "harm"]:
        r_acc = (rules[col].values == df[col].values).mean()
        l_acc = (llm[col].values == df[col].values).mean()
        print(f"{col:14s}  rules: {r_acc:6.1%}   llm: {l_acc:6.1%}")

    print("\n=== failure_mode accuracy by true category (n>=5) ===")
    for cat, g in df.groupby("failure_mode"):
        if len(g) < 5:
            continue
        idx = g.index
        r = (rules.loc[idx, "failure_mode"].values == g["failure_mode"].values).mean()
        l = (llm.loc[idx, "failure_mode"].values == g["failure_mode"].values).mean()
        print(f"  {cat:22s} n={len(g):3d}  rules: {r:5.1%}  llm: {l:5.1%}")

    out = df.copy()
    out["rules_failure_mode"] = rules["failure_mode"]
    out["llm_failure_mode"] = llm["failure_mode"]
    dis = out[(out["rules_failure_mode"] != out["failure_mode"])
              | (out["llm_failure_mode"] != out["failure_mode"])]
    dis.to_csv("data/eval_disagreements.csv", index=False)
    print(f"\n{len(dis)} rows where at least one classifier missed "
          f"-> data/eval_disagreements.csv")


if __name__ == "__main__":
    main()