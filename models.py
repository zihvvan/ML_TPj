import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import lightgbm as lgb
import xgboost as xgb
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

def data_preprocessing():
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

def random_forest_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test) #

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred)
    st.write(accuracy)

def lightGBM_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    lgb_model = lgb.LGBMClassifier(n_estimators=15, random_state=42)
    lgb_model.fit(X_train, y_train)
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_test)

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred)
    st.write(accuracy)

def xgBoost_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)


    # 훈련 및 검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 하이퍼 파라미터 설정
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    # 모델 훈련
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'validation')], early_stopping_rounds=10)

    # 검증 데이터 예측
    y_pred = model.predict(dvalid)

    # 정확도 계산
    acc = accuracy_score(y_valid, [1 if i >= 0.5 else 0 for i in y_pred])
    st.write(accuracy)