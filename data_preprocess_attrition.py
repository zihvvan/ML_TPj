import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
import lightgbm as lgb
from xgboost import XGBClassifier
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
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import pre_process

def make_dummies(df):
    X = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    X = X.drop('Attrition', axis=1)
    y = df['Attrition'].apply(lambda x : 1 if x == "Yes" else 0)
    return X,y

def split_dataset_score(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
    return X_train, X_test, y_train, y_test

def random_forest_score(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(n_estimators=15, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred) # 실제 값, 예측 값 # MAE
    rmse = mean_squared_error(y_test, test_pred, squared=False) # RMSE
    r2 = r2_score(y_test, test_pred)
    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred)
    train_score = model.score(X_train, y_train)
    test_score= model.score(X_test, y_test)


    index = ["RandomForest"]
    total_info = {"정확도": accuracy, "Train점수": train_score, "Test점수": test_score}
    total_df = pd.DataFrame([total_info], index=index)

    st.write(total_df)

def lightgbm_score(X_train, X_test, y_train, y_test):

    lgb_model = lgb.LGBMClassifier(n_estimators=15, random_state=42,  num_iterations= 50) 
    lgb_model.fit(X_train, y_train)
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred) # 실제 값, 예측 값 # MAE
    rmse = mean_squared_error(y_test, test_pred, squared=False) # RMSE
    r2 = r2_score(y_test, test_pred)
    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred)
    train_score = lgb_model.score(X_train, y_train)
    test_score= lgb_model.score(X_test, y_test)


    index = ["LightGBM"]
    total_info = {"정확도": accuracy, "Train점수": train_score, "Test점수": test_score}
    total_df = pd.DataFrame([total_info], index=index)

    st.write(total_df)

def xgBoost_score(X_train, X_valid, y_train, y_valid):

    model_path = "Data/pkl/XGBoost.pkl"
    xgb_model = joblib.load(model_path)

    X_train, X_valid, y_train, y_valid = X_train, X_valid, y_train, y_valid

    y_pred = xgb_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    # 정확도를 계산하여 모델의 성능을 평가합니다.
    train_score = xgb_model.score(X_train, y_train)
    test_score= xgb_model.score(X_valid, y_valid)


    index = ["xgBoost"]
    total_info = {"정확도": accuracy, "Train점수": train_score, "Test점수": test_score}
    total_df = pd.DataFrame([total_info], index=index)

    st.write(total_df)