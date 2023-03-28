import streamlit as st
import pandas as pd
import polynomial_process, visualization_process, models
import pre_process

def view_model1():
    st.title("다중선형회귀 vs 다항선형회귀(Lasso)")
    tab1, tab2, tab3 = st.tabs(["성적 예측","데이터셋 지표분석","시각화"])
    df = pre_process.load_data(1)
    s_df, comparison = polynomial_process.poly_model(df)
    with tab1:
            st.header("성적 예측 모델")
            models.linear_model()
    with tab2:
            st.header("원본 데이터")
            st.write(df)
            st.header("데이터셋 전처리 후")
            processed_df = pre_process.pre_processing(df)
            visualization_process.describe_linear_model(df)
            st.header("One-Hot Enconding")
            st.write(processed_df)
            st.header("Min-Max Scaling")
    with tab3:
            st.header("시각화")
            visualization_process.visualization(df, s_df, comparison)



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
            