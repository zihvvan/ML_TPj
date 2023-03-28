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
import pre_process


def describe_linear_model(df):
    st.write(df.describe(include='all'))

def info_linear_model(df):
    info = df.info()
    info_df = pd.DataFrame(info,index[0])
    st.write(info_df)