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
import model1, model2
import pre_process
import view

def main():
    image1 = Image.open('image/m_img.png')
    st.image(image1, width=600)

    add_selectbox = st.sidebar.selectbox(
        "모델을 선택하세요. ",
        ("성적 예측 모델", "회사퇴사 예측 모델")
    )

    if add_selectbox == "성적 예측 모델":
        view.view_model1()
    else:
        view.view_model2()


main()