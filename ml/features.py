import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

IN_PATH = os.path.join("ml", "ddi_with_smiles.csv")
OUT_DIR = "ml"

print("Loading dataset...")
df = pd.read_csv(IN_PATH)
print(f"Full dataset: {len(df)} pairs")

df = df.sample(n=100000, random_state=42).reset_index(drop=True)
print(f"Sampled down to: {len(df)} pairs")

def smiles_to_fp(smiles, radius, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        return list(fp)
    except:
        return None

y = df["label"].values
np.save(os.path.join(OUT_DIR, "y.npy"), y)
print(f"Labels saved. Shape: {y.shape}")

for radius in [1, 2, 3]:
    print(f"\nGenerating fingerprints at radius={radius}...")
    fps1 = [smiles_to_fp(s, radius) for s in df["drug1_smiles"]]
    fps2 = [smiles_to_fp(s, radius) for s in df["drug2_smiles"]]

    valid = [i for i in range(len(fps1)) if fps1[i] is not None and fps2[i] is not None]
    X = np.array([fps1[i] + fps2[i] for i in valid])

    out_path = os.path.join(OUT_DIR, f"X_r{radius}.npy")
    np.save(out_path, X)
    print(f"  Saved {out_path} — shape: {X.shape}")

print("\nFeature engineering complete!")