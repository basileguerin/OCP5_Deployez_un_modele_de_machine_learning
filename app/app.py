import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "classifier_employee.pkl"

obj = joblib.load(MODEL_PATH)
classifier = obj["model"]
threshold = obj["seuil"]
scaler = obj["scaler"]


if __name__ == '__main__':
    print(MODEL_PATH)
    print(classifier)
    print(threshold)
    print(scaler)