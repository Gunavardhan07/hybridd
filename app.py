import streamlit as st
import torch
from model.model import BioHybrid
from model.utils import smiles_to_graph, compute_descriptors, draw_molecule

st.set_page_config(page_title="BioHybrid", layout="wide")
st.title("🧬 BioHybrid – Drug Interaction & Analysis")

# ---------------- Session State ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Tabs ---------------- #
tabs = st.tabs(["Single Prediction", "Batch Prediction", "Drug Analysis"])

# ---------------- Tab 1: Single Prediction ---------------- #
with tabs[0]:
    st.header("Single Pair Interaction")
    smi1 = st.text_input("Drug 1 SMILES", "CCO")
    smi2 = st.text_input("Drug 2 SMILES", "CCN(CC)CCCC(C)NC1=C2C=CC=CC2=NC=C1")
    if st.button("Predict Interaction Risk"):
        try:
            model = BioHybrid()
            model.eval()
            g1 = smiles_to_graph(smi1)
            g2 = smiles_to_graph(smi2)
            with torch.no_grad():
                prob = model.forward_pair(g1, [smi1], g2, [smi2]).item()
            risk = "Low" if prob<0.33 else "Medium" if prob<0.66 else "High"
            st.success(f"Interaction risk: {prob*100:.2f}% → {risk}")
            st.session_state.history.append((smi1, smi2, prob, risk))
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.session_state.history:
        st.subheader("Prediction History")
        for h in st.session_state.history[-5:]:
            st.write(f"{h[0]} ↔ {h[1]} → {h[2]*100:.2f}% ({h[3]})")

# ---------------- Tab 2: Batch Prediction ---------------- #
with tabs[1]:
    st.header("Batch Prediction CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV with columns: SMILES1,SMILES2", type=["csv"])
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df.head())
        st.warning("Batch prediction not implemented fully in demo.")

# ---------------- Tab 3: Drug Analysis ---------------- #
with tabs[2]:
    st.header("Single Drug Analysis")
    smi = st.text_input("Drug SMILES for analysis", "CCO")
    if st.button("Analyze Drug"):
        try:
            desc = compute_descriptors(smi)
            st.write("Molecular Descriptors:")
            st.json(desc)
            st.image(draw_molecule(smi))
        except Exception as e:
            st.error(f"Error: {str(e)}")
