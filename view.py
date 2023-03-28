import streamlit as st
import pandas as pd
import model1, model2
import pre_process

def view_model1():
    st.title("다중선형회귀 vs 다항선형회귀")
    tab1, tab2, tab3 = st.tabs(["성적 예측","LinearRegression 지표분석", "PolynomialRegression 지표분석"])
    df = pre_process.load_data(1)
    with tab1:
            st.header("성적 예측 모델")
            model2.linear_model()
    with tab2:
            st.header("LinearRegression")
            st.write("## Describe()")
            model1.poly_model(df)
    with tab3:
            st.header("PolynomialRegression")
            st.write("## Describe()")
            model1.poly_model(df)


def view_model2():
    st.title("회사퇴사 예측 모델")
    tab1, tab2, tab3 = st.tabs(["RandomForest", "XGBoost", 'LightGBM'])
    df = pre_process.load_data(2)
    with tab1:
            # RandomForest Model
            model2.random_forest_model(df)
    with tab2:
            # XGBoost Model
            model2.xgBoost_model(df)
    with tab3:
            # LightGBM Model
            model2.lightGBM_model(df)
            