# DrugGuard 💊
### ML-based Drug-Drug Interaction Prediction from Molecular Fingerprints

> Predicting whether two drugs will interact — using only their molecular structures — trained on DrugBank's full interaction database.

---

## Overview

Drug-drug interactions (DDIs) are a major cause of adverse drug events. Testing every possible drug pair in a lab is infeasible — there are millions of combinations. **DrugGuard** takes a computational approach: given two drugs, it predicts whether they are likely to interact, using machine learning on molecular structure data alone.

The pipeline goes from raw DrugBank XML → SMILES strings → Morgan fingerprints → XGBoost classifier, with no external labels or biological pathway data needed.

---

## Dataset

- **Source:** [DrugBank](https://go.drugbank.com/) (full database XML + structures SDF)
- **Positive pairs:** 2.9 million known DDI pairs extracted from DrugBank
- **After SMILES merge:** 2.36 million clean pairs (rows with missing structures dropped)
- **Training sample:** 100,000 positive pairs + 50,000 randomly generated negative pairs (assumed non-interacting)

---

## Methods

### Step 1 — Extracting DDI Pairs (`extract_ddi.py`)
Used `iterparse()` to stream through DrugBank's XML drug-by-drug. For each drug, extracted its DrugBank ID and all listed interaction partners + descriptions. Output: 2.9M pairs with `label = 1`.

### Step 2 — Adding SMILES Strings (`preprocess.py`)
Molecular structures were stored separately in `structures.sdf`. Used **RDKit** to build a DrugBank ID → SMILES mapping, then merged it into the interaction pairs. Dropped rows with missing structures, leaving 2.36M clean pairs.

### Step 3 — Morgan Fingerprints (`features.py`)
Converted each SMILES string into a **2048-bit Morgan fingerprint** — a binary vector encoding the presence/absence of structural patterns around each atom, at a given neighbourhood radius.

- Ran at **radius = 1, 2, and 3** to test how much structural context helps
- Concatenated both drugs' fingerprints → **4096-bit feature vector** per pair
- Saved as `X_r1.npy`, `X_r2.npy`, `X_r3.npy`

### Step 4 — Training (`train.py`)
Trained one **XGBoost** classifier per radius on an 80/20 train-test split.

---

## Results

| Radius | Accuracy | Recall (interaction) | ROC-AUC |
|--------|----------|----------------------|---------|
| r = 1  | 68%      | 0.70                 | 0.753   |
| r = 2  | 70%      | 0.73                 | 0.768   |
| r = 3  | 71%      | 0.73                 | **0.780** |

**Key finding:** Contrary to the initial hypothesis, r=3 outperforms r=1 and r=2. A wider molecular neighbourhood (capturing atoms further from the centre) provides more informative structural context, rather than introducing noise. This suggests that DDI-relevant structural patterns often span more than 1–2 bonds from any given atom.

ROC-AUC of **0.78 using only molecular structure** (no pathway, target, or pharmacokinetic data) is a strong baseline result.

---

## Project Structure

```
DrugGuard/
├── extract_ddi.py      # Parse DrugBank XML → DDI pairs
├── preprocess.py       # Merge SMILES from SDF file
├── features.py         # Generate Morgan fingerprints
├── train.py            # Train & evaluate XGBoost models
├── data/               # (not tracked) Raw DrugBank files
├── X_r1.npy            # Feature matrix at radius 1
├── X_r2.npy            # Feature matrix at radius 2
├── X_r3.npy            # Feature matrix at radius 3
└── README.md
```

---

## How to Run

### Prerequisites
```bash
pip install rdkit xgboost scikit-learn pandas numpy
```

### Pipeline (in order)
```bash
python extract_ddi.py       # Step 1: Extract DDI pairs from XML
python preprocess.py        # Step 2: Merge SMILES strings
python features.py          # Step 3: Generate fingerprints
python train.py             # Step 4: Train and evaluate models
```

> **Note:** DrugBank data files (`full database.xml`, `structures.sdf`) require a free academic account at [drugbank.com](https://go.drugbank.com/) and are not included in this repository.

---

## Tech Stack

- **Python** — core language
- **RDKit** — cheminformatics & fingerprint generation
- **XGBoost** — gradient boosted classifier
- **scikit-learn** — train/test split, evaluation metrics
- **NumPy / Pandas** — data handling

---


