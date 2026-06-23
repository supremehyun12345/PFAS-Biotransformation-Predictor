import joblib
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


# -----------------------------
# Load model and feature columns
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("pfas_biotransformation_rf_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns


model, FEATURE_COLUMNS = load_model()


# -----------------------------
# RDKit descriptor calculation
# -----------------------------
def rdkit_features_from_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("RDKit could not parse this SMILES. Please check the structure.")

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def build_model_row(
    smiles: str,
    pfas_subclass: str,
    charge_category: str,
    ionizable: str,
    matrix_type: str,
    redox_broad: str,
    exposure_mode: str,
    perfluoro_carbon: float,
    s_to_l_ratio: float,
) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Molecular descriptors from SMILES
    desc = rdkit_features_from_smiles(smiles)
    for k, v in desc.items():
        if k in row:
            row[k] = v

    # Numeric environmental descriptors
    if "S_to_L_Ratio_percent" in row:
        row["S_to_L_Ratio_percent"] = s_to_l_ratio
    if "Perfluoro_Carbon" in row:
        row["Perfluoro_Carbon"] = perfluoro_carbon

    # One-hot categorical descriptors
    categorical_values = {
        "PFAS_Subclass": pfas_subclass,
        "Charge_Category": charge_category,
        "Ionizable": ionizable,
        "Matrix_Type": matrix_type,
        "Redox_Broad": redox_broad,
        "Exposure_Mode": exposure_mode,
    }

    for base, value in categorical_values.items():
        col = f"{base}_{value}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PFAS Biotransformation Predictor",
    page_icon="🧪",
    layout="centered",
)

st.title("PFAS Biotransformation Predictor (Yoon model v1.1)")
st.caption(
    "Predicts the probability that PFAS biotransformation will be observed "
    "under user-specified environmental conditions."
    " Questions can be sent to h_yoon@berkeley.edu"
)

st.markdown(
    """
    **Note:** This tool is intended for screening and hypothesis generation only.
    Predictions are based on a literature-derived random forest model and should not
    be interpreted as definitive evidence of environmental persistence or transformation.
    """
)

st.divider()

smiles = st.text_area(
    "PFAS SMILES (e.g., 6:2 FTOH: FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCO)",
    value="FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCO",
    help="Enter a valid SMILES string. RDKit descriptors will be calculated automatically.",
)

col1, col2 = st.columns(2)

with col1:
    pfas_subclass = st.selectbox(
        "PFAS subclass",
        [
            "Amidoamine/Amidoammonium",
            "Cl-PFAS",
            "Ether PFAS",
            "Ft alcohol",
            "Ft carboxylate",
            "Ft specialty precursor",
            "Ft sulfonate",
            "PAP precursor",
            "Sulfonamide precursor",
            "Sulfonamidoacetate precursor",
            "Sulfonamidoamine precursor",
            "Sulfonamidoethanol precursor",
            "Other",
        ],
        index=2,
    )

    charge_category = st.selectbox(
        "Charge category",
        [
            "Neutral",
            "Zwitterionic/cationic",
            "Anionic",
        ],
        index=0,
    )

    ionizable = st.selectbox(
        "Ionizable?",
        [
            "Yes",
            "No",
        ],
        index=1,
    )

with col2:
    matrix_type = st.selectbox(
        "Matrix type",
        [
            "Soil/sediment",
            "No added solids",
            "Activated sludge",
        ],
        index=0,
    )

    redox_broad = st.selectbox(
        "Redox condition",
        [
            "Anaerobic",
            "Aerobic",
        ],
        index=1,
    )

    exposure_mode = st.selectbox(
        "Exposure mode",
        [
            "Liquid-spiked",
            "Presorbed"
        ],
        index=0,
    )

perfluoro_carbon = st.number_input(
    "Perfluoro carbon",
    min_value=0,
    max_value=100,
    value=8,
    step=1,
)

s_to_l_ratio = st.number_input(
    "Solid-to-liquid ratio (%)",
    min_value=0.0,
    max_value=10000.0,
    value=10.0,
    step=0.1,
)

st.divider()

if st.button("Predict biotransformation probability", type="primary"):
    try:
        X_new = build_model_row(
            smiles=smiles,
            pfas_subclass=pfas_subclass,
            charge_category=charge_category,
            ionizable=ionizable,
            matrix_type=matrix_type,
            redox_broad=redox_broad,
            exposure_mode=exposure_mode,
            perfluoro_carbon=perfluoro_carbon,
            s_to_l_ratio=s_to_l_ratio,
        )

        prob_yes = model.predict_proba(X_new)[0, 1]
        pred = model.predict(X_new)[0]

        st.subheader("Prediction")
        st.metric(
            label="Predicted probability of observed biotransformation",
            value=f"{prob_yes:.2%}",
        )

        if prob_yes >= 0.5:
            st.success("Model classification: Likely biotransformed / Yes")
        else:
            st.warning("Model classification: Less likely biotransformed / No")

        with st.expander("Calculated molecular descriptors"):
            desc = rdkit_features_from_smiles(smiles)
            st.dataframe(pd.DataFrame([desc]).T.rename(columns={0: "Value"}))

        with st.expander("Model input vector"):
            st.dataframe(X_new.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(str(e))


st.divider()
st.caption(
    "Model output depends on the training dataset, descriptor definitions, and "
    "category labels used during model development."
)
