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
    headgroup_type: str,
    ionizable: str,
    matrix_type: str,
    redox_broad: str,
    exposure_mode: str,
    incubation_days: float,
    s_to_l_ratio: float,
) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Molecular descriptors from SMILES
    desc = rdkit_features_from_smiles(smiles)
    for k, v in desc.items():
        if k in row:
            row[k] = v

    # Numeric environmental descriptors
    if "Incubation_Time_days" in row:
        row["Incubation_Time_days"] = incubation_days
    if "S_to_L_Ratio_percent" in row:
        row["S_to_L_Ratio_percent"] = s_to_l_ratio

    # One-hot categorical descriptors
    categorical_values = {
        "PFAS_Subclass": pfas_subclass,
        "Headgroup_Type": headgroup_type,
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

st.title("PFAS Biotransformation Predictor")
st.caption(
    "Predicts the probability that PFAS biotransformation will be observed "
    "under user-specified environmental conditions."
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
    "SMILES",
    value="FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCO",
    help="Enter a valid SMILES string. RDKit descriptors will be calculated automatically.",
)

col1, col2 = st.columns(2)

with col1:
    pfas_subclass = st.selectbox(
        "PFAS subclass",
        [
            "Ether PFAS",
            "Fluorotelomer precursor",
            "Ft alcohol",
            "Ft carboxylate",
            "Ft specialty precursor",
            "Ft sulfonate",
            "PAP precursor",
            "PFCA",
            "Sulfonamide precursor",
            "Sulfonamidoacetate precursor",
            "Sulfonamidoamine precursor",
            "Sulfonamidoethanol precursor",
            "Other / baseline",
        ],
        index=2,
    )

    headgroup_type = st.selectbox(
        "Headgroup type",
        [
            "Neutral",
            "Zwitterionic/cationic",
            "Anionic / other baseline",
        ],
        index=0,
    )

    ionizable = st.selectbox(
        "Ionizable?",
        [
            "Yes",
            "No / baseline",
        ],
        index=1,
    )

with col2:
    matrix_type = st.selectbox(
        "Matrix type",
        [
            "Soil/sediment",
            "No added solids",
            "Activated sludge / other baseline",
        ],
        index=0,
    )

    redox_broad = st.selectbox(
        "Redox condition",
        [
            "Anaerobic",
            "Aerobic / baseline",
        ],
        index=1,
    )

    exposure_mode = st.selectbox(
        "Exposure mode",
        [
            "Liquid spike",
            "Pre-sorbed to solids",
            "Other / baseline",
        ],
        index=0,
    )

incubation_days = st.number_input(
    "Incubation time (days)",
    min_value=0.0,
    max_value=5000.0,
    value=90.0,
    step=1.0,
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
            headgroup_type=headgroup_type,
            ionizable=ionizable,
            matrix_type=matrix_type,
            redox_broad=redox_broad,
            exposure_mode=exposure_mode,
            incubation_days=incubation_days,
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
