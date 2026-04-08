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
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("ml/models/model_r3.pkl")

@st.cache_resource(show_spinner="Loading drug database (first time only, ~1 min)...")
def load_drug_data():
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

    return smiles_map, name_to_id, sorted(set(available))

def get_fingerprint(smiles, radius=3, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.uint8)

st.title("💊 DrugGuard")
st.markdown("#### Drug-Drug Interaction Predictor")
st.markdown("Select two drugs to check if they may interact harmfully.")
st.divider()

model = load_model()
smiles_map, name_to_id, available_drugs = load_drug_data()

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
                    st.markdown(f"""
                    <div class="result-box danger">
                        <h2>⚠️ Interaction Likely</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> may interact harmfully.</p>
                        <p class="confidence">Model confidence: {prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box safe">
                        <h2>✅ No Interaction Detected</h2>
                        <p><b>{drug1}</b> and <b>{drug2}</b> are not predicted to interact.</p>
                        <p class="confidence">Model confidence: {(1-prob)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<p class="disclaimer">⚕️ This is a computational prediction only. Always consult a healthcare professional before combining medications.</p>', unsafe_allow_html=True)

st.divider()
