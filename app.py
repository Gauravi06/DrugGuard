from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
import pickle
import json
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
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 1rem;
    }
    .info-card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .info-card-full {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 12px;
    }
    .info-card-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 6px;
    }
    .info-card-value {
        font-size: 0.92rem;
        color: #e0e0e0;
        line-height: 1.5;
    }
    .drug-pill {
        display: inline-block;
        background-color: #2a2a4a;
        color: #a0a0ff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .warning-pill {
        display: inline-block;
        background-color: #3a1a1a;
        color: #ff9999;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .safe-pill {
        display: inline-block;
        background-color: #1a3a1a;
        color: #99ff99;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

CACHE_FILE = "drug_data_cache.pkl"
GROQ_CACHE_FILE = "ml/groq_cache.json"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("ml/models/model_r3.pkl")


@st.cache_resource(show_spinner="Loading drug database...")
def load_drug_data():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

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

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)

    return result


@st.cache_resource(show_spinner=False)
def load_ddi_descriptions():
    ddi_path = os.path.join("ml", "ddi_descriptions_cache.csv")
    if not os.path.exists(ddi_path):
        return {}
    try:
        df = pd.read_csv(ddi_path)
        desc_map = {}
        for _, row in df.iterrows():
            key  = (row["drug1_id"], row["drug2_id"])
            rkey = (row["drug2_id"], row["drug1_id"])
            if pd.notna(row.get("description")):
                desc_map[key]  = row["description"]
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


def get_ai_explanation(drug1, drug2):
    key = f"{drug1.lower()}|{drug2.lower()}"

    if os.path.exists(GROQ_CACHE_FILE):
        with open(GROQ_CACHE_FILE, "r") as f:
            cache = json.load(f)
        if key in cache:
            return cache[key]
    else:
        cache = {}

    try:
        import requests as req
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        prompt = (
            f"Explain this drug pair in exactly this JSON format with no extra text:\n"
            f"{{\n"
            f'  "drug1_class": "drug class of {drug1}",\n'
            f'  "drug1_use": "what {drug1} is typically used for (one short sentence)",\n'
            f'  "drug2_class": "drug class of {drug2}",\n'
            f'  "drug2_use": "what {drug2} is typically used for (one short sentence)",\n'
            f'  "interaction_effect": "what harmful effect combining them causes (one short sentence)",\n'
            f'  "interaction_reason": "why this happens mechanistically (one short sentence)"\n'
            f"}}\n"
            f"Be medically accurate. Return only valid JSON, nothing else."
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        response = req.post(url, json=payload, headers=headers, timeout=15)
        resp_json = response.json()

        if "choices" not in resp_json:
            return None

        raw = resp_json["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        cache[key] = parsed
        with open(GROQ_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        return parsed

    except Exception:
        return None


def render_explanation(drug1, drug2, data, interaction):
    if isinstance(data, str):
        st.markdown(f'<div class="info-card-full"><div class="info-card-value">{data}</div></div>', unsafe_allow_html=True)
        return

    if not isinstance(data, dict):
        return

    pill1 = "warning-pill" if interaction else "safe-pill"
    pill2 = "warning-pill" if interaction else "safe-pill"

    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-label">💊 {drug1}</div>
            <div class="drug-pill">{data.get('drug1_class', 'Unknown class')}</div>
            <div class="info-card-value">{data.get('drug1_use', '')}</div>
        </div>
        <div class="info-card">
            <div class="info-card-label">💊 {drug2}</div>
            <div class="drug-pill">{data.get('drug2_class', 'Unknown class')}</div>
            <div class="info-card-value">{data.get('drug2_use', '')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if interaction:
        st.markdown(f"""
        <div class="info-card-full">
            <div class="info-card-label">⚠️ What happens when combined</div>
            <div class="{'warning-pill'}">Effect</div>
            <div class="info-card-value">{data.get('interaction_effect', '')}</div>
            <br/>
            <div class="{'warning-pill'}">Why</div>
            <div class="info-card-value">{data.get('interaction_reason', '')}</div>
        </div>
        """, unsafe_allow_html=True)


st.title("💊 DrugGuard")
st.markdown("#### Drug-Drug Interaction Predictor")
st.markdown("Select two drugs below to check whether combining them may cause a harmful interaction.")
st.divider()

model = load_model()
smiles_map, name_to_id, available_drugs = load_drug_data()
ddi_descriptions = load_ddi_descriptions()

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

                if pred == 1:
                    severity = severity_label(prob)
                    st.markdown(f"""
                    <div class="result-box danger">
                        <h2>⚠️ Interaction Likely</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> may interact harmfully.</p>
                        <p class="confidence">Severity: <b>{severity}</b> &nbsp;|&nbsp; Model confidence: {prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.spinner("Looking up details..."):
                        db_desc = ddi_descriptions.get((id1, id2))
                        if db_desc:
                            explanation = db_desc
                        else:
                            explanation = get_ai_explanation(drug1, drug2)

                    if explanation:
                        render_explanation(drug1, drug2, explanation, interaction=True)
                    else:
                        st.markdown('<div class="info-card-full"><div class="info-card-value">No specific description available. The model predicted a likely interaction based on molecular structure patterns.</div></div>', unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="result-box safe">
                        <h2>✅ No Interaction Detected</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> are not predicted to interact.</p>
                        <p class="confidence">Model confidence: {(1-prob)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.spinner("Looking up drug details..."):
                        explanation = get_ai_explanation(drug1, drug2)

                    if explanation:
                        render_explanation(drug1, drug2, explanation, interaction=False)

                st.markdown(
                    '<p class="disclaimer">This is a computational prediction only. '
                    'Always consult a healthcare professional before combining medications.</p>',
                    unsafe_allow_html=True
                )

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

