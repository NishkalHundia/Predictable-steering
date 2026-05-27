import pandas as pd, numpy as np

df = pd.read_csv('results/open_ended_projection_link/gemma-2-9b-it/myopic-reward/per_prompt_results.csv')
print("Columns:", list(df.columns))
print()

bscore_cols = [c for c in df.columns if 'behavior_score' in c or 'fluency_score' in c]
print("Score cols:", bscore_cols[:10])
print()

for c in bscore_cols[:6]:
    vals = df[c]
    non_null = vals.dropna()
    print(f"{c}: {len(non_null)}/{len(vals)} non-null, sample: {non_null.head(5).tolist()}")

print()
print("=== behavior_score_0 value distribution ===")
if 'behavior_score_0' in df.columns:
    print(df['behavior_score_0'].value_counts(dropna=False).head(20))
