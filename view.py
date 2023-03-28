import streamlit as st
import pandas as pd
import polynomial_process, models
import pre_process

def view_model1():
    st.title("다중선형회귀 vs 다항선형회귀")
    tab1, tab2, tab3, tab4 = st.tabs(["성적 예측","데이터셋 전처리","LinearRegression 지표분석", "PolynomialRegression 지표분석"])
    df = pre_process.load_data(1)
    with tab1:
            st.header("성적 예측 모델")
            models.linear_model()
    with tab2:
            st.header("데이터셋 전처리 전")
            df = pre_process.load_data()
            st.write(df)
            st.header("데이터셋 전처리 후")
            processed_df = pre_process.pre_processing(df)
            st.write(processed_df)
    with tab3:
            st.header("LinearRegression")
            st.write("## Describe()")
            polynomial_process.poly_model(df)
    with tab4:
            st.header("PolynomialRegression")
            st.write("## Describe()")
            polynomial_process.poly_model(df)


def view_model2():
    st.title("회사퇴사 예측 모델")
    tab1, tab2, tab3 = st.tabs(["RandomForest", "XGBoost", 'LightGBM'])
    df = pre_process.load_data(2)
    with tab1:
            # RandomForest Model
            models.random_forest_model(df)
    with tab2:
            # XGBoost Model
            models.xgBoost_model(df)
    with tab3:
            # LightGBM Model
            models.lightGBM_model(df)
            