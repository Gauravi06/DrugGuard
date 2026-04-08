import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

st.set_page_config(
    page_title="DrugGuard",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
<style>
    .result-box {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
    }
    .danger {
        background-color: #FCEBEB;
        border: 1.5px solid #F09595;
        color: #791F1F;
    }
    .safe {
        background-color: #EAF3DE;
        border: 1.5px solid #97C459;
        color: #27500A;
    }
    .moderate {
        background-color: #FAEEDA;
        border: 1.5px solid #EF9F27;
        color: #633806;
    }
    .confidence {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 1rem;
    }
    .explanation-box {
        background-color: var(--background-color);
        border-left: 4px solid #F09595;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("ml/models/model_r3.pkl")

@st.cache_resource(show_spinner="Loading drug database...")
def load_drug_data():
    cache_path = "ml/drug_smiles_cache.csv"
    df = pd.read_csv(cache_path)
    smiles_map = dict(zip(df["DrugBank ID"], df["SMILES"]))
    name_to_id = dict(zip(df["Common name"].str.lower(), df["DrugBank ID"]))
    available = sorted(df["Common name"].dropna().unique().tolist())
    return smiles_map, name_to_id, available

@st.cache_resource(show_spinner=False)
def load_ddi_descriptions():
    ddi = pd.read_csv("ml/ddi_pairs.csv")
    desc_map = {}
    for _, row in ddi.iterrows():
        key = (row["drug1_id"], row["drug2_id"])
        desc_map[key] = row.get("description", "")
        key2 = (row["drug2_id"], row["drug1_id"])
        desc_map[key2] = row.get("description", "")
    return desc_map

def get_fingerprint(smiles, radius=3, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.uint8)

def get_severity(prob):
    if prob >= 0.75:
        return "High", "danger"
    elif prob >= 0.55:
        return "Moderate", "moderate"
    else:
        return "Low", "safe"

model = load_model()
smiles_map, name_to_id, available_drugs = load_drug_data()
desc_map = load_ddi_descriptions()

st.title("💊 DrugGuard")
st.markdown("#### Drug-Drug Interaction Predictor")
st.markdown("Select two drugs below to check whether combining them may cause a harmful interaction.")
st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<p class="stat-label">Model</p><p class="stat-value">XGBoost</p>', unsafe_allow_html=True)
with c2:
    st.markdown('<p class="stat-label">Fingerprint</p><p class="stat-value">Morgan r=3</p>', unsafe_allow_html=True)
with c3:
    st.markdown('<p class="stat-label">ROC-AUC</p><p class="stat-value">0.7797</p>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("First drug", options=["Select a drug..."] + available_drugs)
with col2:
    drug2 = st.selectbox("Second drug", options=["Select a drug..."] + available_drugs)

if st.button("Check Interaction", type="primary", use_container_width=True):
    if drug1 == "Select a drug..." or drug2 == "Select a drug...":
        st.warning("Please select both drugs.")
    elif drug1 == drug2:
        st.warning("Please select two different drugs.")
    else:
        id1 = name_to_id.get(drug1.lower())
        id2 = name_to_id.get(drug2.lower())
        smi1 = smiles_map.get(id1)
        smi2 = smiles_map.get(id2)

        if not smi1 or not smi2:
            st.error("Molecular structure not found for one or both drugs.")
        else:
            fp1 = get_fingerprint(smi1)
            fp2 = get_fingerprint(smi2)

            if fp1 is None or fp2 is None:
                st.error("Could not generate fingerprint for one or both drugs.")
            else:
                X = np.concatenate([fp1, fp2]).reshape(1, -1)
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][1]
                severity, style = get_severity(prob)

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-box {style}">
                        <h2>⚠️ Interaction Likely</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> may interact harmfully.</p>
                        <p class="confidence">Severity: <b>{severity}</b> &nbsp;|&nbsp; Model confidence: <b>{prob*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    desc = desc_map.get((id1, id2), "") or desc_map.get((id2, id1), "")
                    if desc:
                        st.markdown("**What this interaction means:**")
                        st.markdown(f"""
                        <div class="explanation-box">
                            {desc}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="explanation-box">
                            No specific interaction description available in the database for this pair.
                            The model has predicted a likely interaction based on molecular structure patterns.
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="result-box safe">
                        <h2>✅ No Interaction Detected</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> are not predicted to interact.</p>
                        <p class="confidence">Model confidence: <b>{(1-prob)*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<p class="disclaimer">⚕️ This is a computational prediction only. Always consult a healthcare professional before combining medications.</p>', unsafe_allow_html=True)

st.divider()

with st.expander("About DrugGuard"):
    st.markdown("""
    **DrugGuard** is an academic ML project built at Vishwakarma Institute of Technology.

    It predicts drug-drug interactions using:
    - **Morgan Fingerprints** — numerical representations of each drug's molecular structure, generated using RDKit
    - **XGBoost classifier** — trained on 100,000 drug pairs from DrugBank 6.0
    - **Radius experiment** — we tested r=1, r=2, r=3 and found r=3 performs best (ROC-AUC: 0.7797)

    | Radius | Accuracy | Recall | ROC-AUC |
    |--------|----------|--------|---------|
    | r=1 | 68% | 0.70 | 0.753 |
    | r=2 | 70% | 0.73 | 0.768 |
    | r=3 | 71% | 0.73 | 0.780 |

    **Dataset:** DrugBank 6.0 (academic license) — 1.4M known drug interactions
    """)

