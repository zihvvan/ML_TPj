import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
# import lightgbm as lgb
from xgboost import XGBClassifier
from PIL import Image
from math import sqrt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

def data_preprocessing(df):
    df1 = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
    df2 = df1.loc[:,['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked','OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager','JobRole','Attrition']]

    df2.info()

    X = df2.iloc[:,:-1].values
    cy = df2.iloc[:,-1:].values   
    y = np.array([1 if x[0] == "Yes" else 0 for x in cy])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [1,3,6,8,13,17,29])], remainder='passthrough')
    X = ct.fit_transform(X)
    return X,y

def random_forest_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    with st.echo(code_location="below"):
        model_path = "Data/pkl/RandomForest_model.pkl"
        model = joblib.load(model_path)
        st.write("## Randomforest model")
    
    train_pred_dt = model.predict(X_train) 
    test_pred_dt = model.predict(X_test)
    
    predict_button_dt1 = st.button('예측')

    if predict_button_dt1:        
        st.write(f'Train-set : {model.score(X_train, y_train)}')
        st.write(f'Test-set : {model.score(X_test, y_test)}')

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred_dt)
    st.write(accuracy)

def lightGBM_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    with st.echo(code_location="below"):
        model_path = "Data/pkl/LightGBM_model.pkl"
        model = joblib.load(model_path)
        st.write("## lightGBM_model model")
    
    train_pred_dt = model.predict(X_train) 
    test_pred_dt = model.predict(X_test)
    
    predict_button_dt2 = st.button('예측하기')

    if predict_button_dt2:        
        st.write(f'Train-set : {model.score(X_train, y_train)}')
        st.write(f'Test-set : {model.score(X_test, y_test)}')

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred_dt)
    st.write(accuracy)

def xgBoost_model(df):
    with st.echo(code_location="below"):
        model_path = "Data/pkl/XGBoost_model.pkl"
        xgb = joblib.load(model_path)
        st.write("## XGBoost_model model")

    X,y = data_preprocessing(df)
    # 훈련 및 검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # train_pred_dt = xgb.predict(X_train) 
    # test_pred_dt = xgb.predict(X_valid)
    y_pred = xgb_model.predict(X_valid)
    # 정확도 계산

    predict_button_dt3 = st.button('예측하기')

    if predict_button_dt3:        
        st.write(f'Train-set : {xgb_model.score(X_train, y_train)}')
        st.write(f'Test-set : {xgb_model.score(X_valid, y_valid)}')

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_valid, y_pred)
    st.write(accuracy)