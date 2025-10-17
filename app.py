import streamlit as st
import torch
from model.model import BioHybrid
from model.utils import smiles_to_graph, compute_descriptors, draw_molecule

st.set_page_config(page_title="BioHybrid", layout="wide")
st.title("🧬 BioHybrid – Drug Interaction & Analysis")

if "history" not in st.session_state:
    st.session_state.history = []

tabs = st.tabs(["Single Prediction","Batch Prediction","Drug Analysis"])

with tabs[0]:
    st.header("Single Pair Interaction")
    smi1 = st.text_input("Drug 1 SMILES", "CCO")
    smi2 = st.text_input("Drug 2 SMILES", "CCN(CC)CCCC(C)NC1=C2C=CC=CC2=NC=C1")
    if st.button("Predict Interaction"):
        try:
            model = BioHybrid()
            model.eval()
            g1, g2 = smiles_to_graph(smi1), smiles_to_graph(smi2)
            with torch.no_grad():
                prob = model.forward_pair(g1,[smi1],g2,[smi2]).item()
            risk = "Low" if prob<0.33 else "Medium" if prob<0.66 else "High"
            st.success(f"Risk: {prob*100:.2f}% → {risk}")
            st.session_state.history.append((smi1,smi2,prob,risk))
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.session_state.history:
        st.subheader("History")
        for h in st.session_state.history[-5:]:
            st.write(f"{h[0]} ↔ {h[1]} → {h[2]*100:.2f}% ({h[3]})")

with tabs[1]:
    st.header("Batch Prediction")
    uploaded = st.file_uploader("CSV with SMILES1,SMILES2", type=["csv"])
    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded)
        st.write(df.head())
        st.warning("Batch prediction demo only.")

with tabs[2]:
    st.header("Single Drug Analysis")
    smi = st.text_input("Drug SMILES for analysis","CCO")
    if st.button("Analyze Drug"):
        try:
            desc = compute_descriptors(smi)
            st.json(desc)
            st.image(draw_molecule(smi))
        except Exception as e:
            st.error(f"Error: {str(e)}")
