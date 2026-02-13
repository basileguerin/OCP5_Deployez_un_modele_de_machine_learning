import streamlit as st
import requests

st.set_page_config(page_title="HRPredict", layout="centered")
st.title("HRPredict")

API_BASE = st.secrets.get("API_BASE")

# Récupérer la liste des features depuis l'API
meta = requests.get(f"{API_BASE}/metadata").json()
features_order = meta["features_order"]

st.subheader("Input features")

features = {}
for name in features_order:
    features[name] = st.number_input(name, value=0.0)

if st.button("Predict"):
    r = requests.post(f"{API_BASE}/predict", json={"features": features})

    if r.status_code == 200:
        res = r.json()
        st.success("Prediction done ✅")
        st.write("Request ID:", res.get("request_id"))
        st.write("Probability:", res["probability"])
        st.write("Prediction:", res["prediction"])
        st.write("Threshold:", res["threshold"])
    else:
        st.error(f"API error {r.status_code}")
        st.code(r.text)
