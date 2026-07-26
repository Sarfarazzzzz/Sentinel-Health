import pandas as pd
df = pd.read_parquet("data/silver/events_dedup.parquet")

# Which raw spellings got merged into each canonical name?
g = df.groupby("manufacturer")["manufacturer_raw"].unique()
big = g[g.map(len) > 1].head(30)
for canon, variants in big.items():
    print(canon, "<-", list(variants)[:6])


#%%

import pandas as pd
df = pd.read_parquet("data/silver/events.parquet")
empty = df["narrative"].str.len().fillna(0) == 0
print(f"empty narratives: {empty.sum():,} ({empty.mean():.1%})")

#%%

import pandas as pd
df = pd.read_parquet("data/silver/events.parquet")

# 1. Did ingestion itself duplicate reports? (pagination instability would show here)
print(f"rows: {len(df):,}   unique report_keys: {df['report_key'].nunique():,}")

# 2. What are the most-repeated narratives?
top = df["narrative_hash"].value_counts().head(10)
for h, count in top.items():
    text = df.loc[df["narrative_hash"] == h, "narrative"].iloc[0]
    print(f"\n{count:,} copies: {text[:200]}")

#%%

df = pd.read_parquet("data/gold/events_extracted.parquet")
print(df["failure_mode"].value_counts())

# Read 40 narratives the classifier couldn't categorize:
unknown = df[df["failure_mode"] == "unknown"]["narrative"]
for t in unknown.sample(40, random_state=1):
    print("---", t[:300])