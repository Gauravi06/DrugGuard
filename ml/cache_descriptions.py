import pandas as pd

print("Loading DDI pairs...")
ddi = pd.read_csv("ml/ddi_pairs.csv")
print(f"Loaded {len(ddi)} pairs")

ddi = ddi[["drug1_id", "drug2_id", "description"]].dropna(subset=["description"])
ddi.to_csv("ml/ddi_descriptions_cache.csv", index=False)
print(f"Saved {len(ddi)} descriptions to ml/ddi_descriptions_cache.csv")