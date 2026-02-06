import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path

st.set_page_config(page_title="HRPredict", layout="centered")

st.title("HRPredict")
st.subheader("Model loading test")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "classifier_employee.pkl"

# Chargement du modèle
obj = joblib.load(MODEL_PATH)
classifier = obj["model"]
threshold = obj["seuil"]
scaler = obj["scaler"]

st.success("Model loaded successfully")

st.write("### Model")
st.write(classifier)

st.write("### Decision threshold")
st.write(threshold)

st.write("### Scaler")
st.write(scaler)

st.caption(f"Model path: {MODEL_PATH}")
