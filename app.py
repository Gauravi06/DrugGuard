from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
import pickle
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
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .danger {
        background-color: #FCEBEB;
        border: 1px solid #F09595;
        color: #791F1F;
    }
    .safe {
        background-color: #EAF3DE;
        border: 1px solid #97C459;
        color: #27500A;
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
        background-color: #1e1e1e;
        border-left: 3px solid #F09595;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: #ddd;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


CACHE_FILE = "drug_data_cache.pkl"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("ml/models/model_r3.pkl")


@st.cache_resource(show_spinner="Loading drug database...")
def load_drug_data():
    # Use disk cache if available — makes restarts near-instant
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    # First-time only: parse the large SDF file (~1 min)
    sdf_path = os.path.join("backend", "data", "structures.sdf")
    voc_path = os.path.join("backend", "data", "drugbank vocabulary.csv")

    voc = pd.read_csv(voc_path)

    suppl = Chem.SDMolSupplier(sdf_path)
    smiles_map = {}
    for mol in suppl:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        did = props.get("DRUGBANK_ID")
        if did:
            smiles_map[did] = Chem.MolToSmiles(mol)

    voc = voc[voc["DrugBank ID"].isin(smiles_map.keys())]
    voc = voc.dropna(subset=["Common name"])
    voc["Common name"] = voc["Common name"].str.strip()

    name_to_id = dict(zip(voc["Common name"].str.lower(), voc["DrugBank ID"]))

    def is_common_name(name):
        if not isinstance(name, str):
            return False
        if name.startswith(("+", "-", "(", "[")):
            return False
        if any(char.isdigit() for char in name[:3]):
            return False
        if name.isupper():
            return False
        if len(name) > 40:
            return False
        return True

    available = [
        row["Common name"] for _, row in voc.iterrows()
        if is_common_name(row["Common name"])
        and row["DrugBank ID"] in smiles_map
    ]

    result = smiles_map, name_to_id, sorted(set(available))

    # Save to disk so next restart is instant
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)

    return result


@st.cache_data(show_spinner=False)
def load_ddi_descriptions():
    ddi_path = os.path.join("ml", "ddi_with_smiles.csv")
    if not os.path.exists(ddi_path):
        return {}
    try:
        df = pd.read_csv(ddi_path, usecols=["drug1_id", "drug2_id", "description"])
        desc_map = {}
        for _, row in df.iterrows():
            key = (row["drug1_id"], row["drug2_id"])
            rkey = (row["drug2_id"], row["drug1_id"])
            if pd.notna(row.get("description")):
                desc_map[key] = row["description"]
                desc_map[rkey] = row["description"]
        return desc_map
    except Exception:
        return {}


def get_fingerprint(smiles, radius=3, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.uint8)


def severity_label(prob):
    if prob >= 0.80:
        return "High"
    elif prob >= 0.65:
        return "Moderate"
    else:
        return "Low"


@st.cache_data(show_spinner=False)
def get_ai_explanation(drug1, drug2):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"In 3-4 sentences, briefly explain: "
            f"(1) what {drug1} is typically used for, "
            f"(2) what {drug2} is typically used for, "
            f"(3) why and how combining these two drugs may cause a harmful interaction. "
            f"Be concise, medically accurate, and accessible to a general audience. "
            f"Write as plain flowing text with no bullet points or headers."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except ImportError:
        return None
    except Exception:
        return None


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💊 DrugGuard")
st.markdown("#### Drug-Drug Interaction Predictor")
st.markdown("Select two drugs below to check whether combining them may cause a harmful interaction.")
st.divider()

# ── Load resources ─────────────────────────────────────────────────────────────
model = load_model()
smiles_map, name_to_id, available_drugs = load_drug_data()
ddi_descriptions = load_ddi_descriptions()

# ── Drug selection ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("First drug", options=["Select a drug..."] + available_drugs)
with col2:
    drug2 = st.selectbox("Second drug", options=["Select a drug..."] + available_drugs)

# ── Predict ────────────────────────────────────────────────────────────────────
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

                if pred == 1:
                    severity = severity_label(prob)
                    st.markdown(f"""
                    <div class="result-box danger">
                        <h2>⚠️ Interaction Likely</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> may interact harmfully.</p>
                        <p class="confidence">Severity: <b>{severity}</b> &nbsp;|&nbsp; Model confidence: {prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    db_desc = ddi_descriptions.get((id1, id2))
                    if db_desc:
                        explanation = db_desc
                    else:
                        with st.spinner("Looking up interaction details..."):
                            explanation = get_ai_explanation(drug1, drug2)

                    if explanation:
                        st.markdown(
                            f'<div class="explanation-box">{explanation}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="explanation-box">No specific description available for this pair. '
                            'The model has predicted a likely interaction based on molecular structure patterns.</div>',
                            unsafe_allow_html=True
                        )

                else:
                    st.markdown(f"""
                    <div class="result-box safe">
                        <h2>✅ No Interaction Detected</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> are not predicted to interact.</p>
                        <p class="confidence">Model confidence: {(1-prob)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    '<p class="disclaimer">⚕️ This is a computational prediction only. '
                    'Always consult a healthcare professional before combining medications.</p>',
                    unsafe_allow_html=True
                )

# ── About ──────────────────────────────────────────────────────────────────────
st.divider()

with st.expander("About DrugGuard"):
    st.markdown("""
**DrugGuard** predicts drug-drug interactions using:

- **Morgan Fingerprints** — numerical representations of each drug's molecular structure, generated using RDKit
- **XGBoost classifier** — trained on 100,000 drug pairs from DrugBank 6.0
- **Radius experiment** — we tested r=1, r=2, r=3 and found r=3 performs best (ROC-AUC: 0.7797)
""")

    st.table(pd.DataFrame({
        "Radius": ["r=1", "r=2", "r=3"],
        "Accuracy": ["68%", "70%", "71%"],
        "Recall": [0.70, 0.73, 0.73],
        "ROC-AUC": [0.753, 0.768, 0.780],
    }))

    st.markdown("**Dataset:** DrugBank 6.0 (academic license) — 1.4M known drug interactions")