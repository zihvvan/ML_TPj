import streamlit as st
import numpy as np
import pandas as pd

def make_dummies(df):
    X = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    X = X.drop('Attrition', axis=1)
    y = df['Attrition'].apply(lambda x : 1 if x == "Yes" else 0)
    return X,y

def create_table():
    model_score_data = {
        "Random_Forest" : 0.9863945578231292,
        "Grid_Random_Forest" : 0.9863945578231292,
        "LightGBM" : 0.9931972789115646,
        "Grid_LightGBM" : 0.9183673469387755,
        "XGBoost" : 0.8639455782312925,
        "GridXGBoost" : 0.8673469387755102
    }
    index = "성능"
    model_scores = pd.DataFrame([model_score_data], index_col=index)
    st.write(model_scores)
