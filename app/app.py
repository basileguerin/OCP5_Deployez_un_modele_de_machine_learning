import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = 'model/classifier_employee.pkl'
model = joblib.load(MODEL_PATH)['model']
threshold = joblib.load(MODEL_PATH)['seuil']


if __name__ == '__main__':
    print(model)
    print(threshold)