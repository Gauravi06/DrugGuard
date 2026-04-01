import pandas as pd
import os
from rdkit import Chem

DDI_PATH = os.path.join("ml", "ddi_pairs.csv")
SDF_PATH = os.path.join("backend", "data", "structures.sdf")
VOC_PATH = os.path.join("backend", "data", "drugbank vocabulary.csv")
OUT_PATH = os.path.join("ml", "ddi_with_smiles.csv")

print("Extracting SMILES from structures.sdf...")
suppl = Chem.SDMolSupplier(SDF_PATH)

smiles_map = {}
for mol in suppl:
    if mol is None:
        continue
    drugbank_id = mol.GetPropsAsDict().get("DRUGBANK_ID", None)
    if drugbank_id:
        smiles_map[drugbank_id] = Chem.MolToSmiles(mol)

print(f"Extracted SMILES for {len(smiles_map)} drugs")

print("Loading vocabulary for drug names...")
voc = pd.read_csv(VOC_PATH)
name_map = voc.set_index("DrugBank ID")["Common name"].to_dict()

print("Loading DDI pairs...")
ddi = pd.read_csv(DDI_PATH)
print(f"DDI pairs loaded: {len(ddi)}")

print("Merging...")
ddi["drug1_smiles"] = ddi["drug1_id"].map(smiles_map)
ddi["drug2_smiles"] = ddi["drug2_id"].map(smiles_map)
ddi["drug1_name"]   = ddi["drug1_id"].map(name_map)
ddi["drug2_name"]   = ddi["drug2_id"].map(name_map)

before = len(ddi)
ddi = ddi.dropna(subset=["drug1_smiles", "drug2_smiles"])
after = len(ddi)

print(f"Rows before dropping missing SMILES: {before}")
print(f"Rows after dropping missing SMILES:  {after}")
print(f"Rows dropped: {before - after}")

ddi.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(f"Final dataset: {len(ddi)} drug pairs ready for fingerprinting")
