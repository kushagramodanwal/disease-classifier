import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Symptom Checker", page_icon="🩺")

@st.cache_resource
def load():
    model = joblib.load("model/model.joblib")
    symptoms = json.load(open("model/symptoms.json"))
    return model, symptoms

model, symptoms = load()
pretty = {s: s.replace("_", " ").capitalize() for s in symptoms}

st.title("🩺 Symptom-based disease classifier")
st.warning(
    "Educational demo only. This is not medical advice and cannot diagnose "
    "any condition. See a qualified clinician for health concerns."
)

chosen = st.multiselect(
    "Select the symptoms you have:", symptoms, format_func=lambda s: pretty[s]
)

if st.button("Predict", type="primary", disabled=not chosen):
    x = pd.DataFrame([[1 if s in chosen else 0 for s in symptoms]], columns=symptoms)
    proba = model.predict_proba(x)[0]
    top = np.argsort(proba)[::-1][:3]
    st.subheader("Most likely matches")
    for i in top:
        st.write(f"**{model.classes_[i]}**")
        st.progress(float(proba[i]), text=f"{proba[i]:.0%}")
    if len(chosen) < 3:
        st.info("Few symptoms selected, so predictions will be unreliable.")
