import streamlit as st
import numpy as np
import pandas as pd

def make_dummies(df):
    X = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    X = X.drop('Attrition', axis=1)
    y = df['Attrition'].apply(lambda x : 1 if x == "Yes" else 0)
    return X,y

def create_table():
    lgbm_score_data = {
        'LightGBM' : 0.8503401360544217,
        'Grid_LightGBM' : 0.8707482993197279,
        
    }