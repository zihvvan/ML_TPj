import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import model1, model2, pre_process

def view_model1():

    st.title("다중선형회귀 vs 다항선형회귀")
    tab1, tab2 = st.tabs(["LinearRegression", "Polynomial Regression"])
    df = pre_process.load_data(1)
    with tab1:
            st.header("LinearRegression")
            model1.linear_model()
    with tab2:
            st.header("Polynomial Regression")
            st.write("## 전처리 후 데이터의 모습")
            model1.poly_model(df)


def view_model2():
    st.title("회사퇴사 예측 모델")
    tab1, tab2, tab3 = st.tabs(["RandomForest", "XGBoost", 'LightGBM'])
    df = pre_process.load_data(2)
    with tab1:
            st.header("RandomForest")
            model2.random_forest_model(df)
    with tab2:
            st.header("XGBoost")
            model2.xgBoost_model(df)
    with tab3:
            st.header("LightGBM")
            model2.lightGBM_model(df)
            

def main():
    image1 = Image.open('image/m_img.png')
    st.image(image1, width=600)

    add_selectbox = st.sidebar.selectbox(
        "모델을 선택하세요. ",
        ("성적 예측 모델", "회사퇴사 예측 모델")
    )

    if add_selectbox == "성적 예측 모델":
        view_model1()
    else:
        view_model2()


main()