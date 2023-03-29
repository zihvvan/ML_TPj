import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image
from math import sqrt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import pre_process

def make_dummies(df):
    X = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    X = X.drop('Attrition', axis=1)
    y = df['Attrition'].apply(lambda x : 1 if x == "Yes" else 0)