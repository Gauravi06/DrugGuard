import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

OUT_DIR = "ml/models"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv("ml/ddi_with_smiles.csv")

interacting_pairs = set(zip(df["drug1_id"], df["drug2_id"]))
unique_drugs = df[["drug1_id","drug1_smiles"]].drop_duplicates().rename(
    columns={"drug1_id":"drug_id","drug1_smiles":"smiles"})

print(f"Unique drugs available: {len(unique_drugs)}")

print("Generating true negative pairs...")
rng = np.random.default_rng(42)
drug_ids   = unique_drugs["drug_id"].values
drug_smi   = unique_drugs["smiles"].values
neg_s1, neg_s2 = [], []

attempts = 0
while len(neg_s1) < 50000 and attempts < 1000000:
    i, j = rng.integers(0, len(drug_ids), size=2)
    if i != j and (drug_ids[i], drug_ids[j]) not in interacting_pairs:
        neg_s1.append(drug_smi[i])
        neg_s2.append(drug_smi[j])
    attempts += 1

print(f"Generated {len(neg_s1)} negative pairs")

pos = df.sample(n=50000, random_state=42).reset_index(drop=True)
pos_s1 = pos["drug1_smiles"].tolist()
pos_s2 = pos["drug2_smiles"].tolist()

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

def smiles_to_fp(smiles, radius, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        return np.array(fp, dtype=np.uint8)
    except:
        return None

all_s1 = pos_s1 + neg_s1
all_s2 = pos_s2 + neg_s2
y_all  = np.array([1]*len(pos_s1) + [0]*len(neg_s1))

for radius in [1, 2, 3]:
    print(f"\n{'='*40}")
    print(f"Radius = {radius}")
    print(f"{'='*40}")

    print("Generating fingerprints...")
    fps1 = [smiles_to_fp(s, radius) for s in all_s1]
    fps2 = [smiles_to_fp(s, radius) for s in all_s2]
    valid = [i for i in range(len(fps1)) if fps1[i] is not None and fps2[i] is not None]
    X = np.array([np.concatenate([fps1[i], fps2[i]]) for i in valid], dtype=np.uint8)
    y = y_all[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training...")
    model = XGBClassifier(
        n_estimators=100, max_depth=6,
        learning_rate=0.1, random_state=42,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["No interaction","Interaction"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, os.path.join(OUT_DIR, f"model_r{radius}.pkl"))
    print(f"Model saved: model_r{radius}.pkl")

print("\nAll 3 models trained and saved!")
