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
    df1 = df.drop(['BusinessTravel','Department','Education','EducationField','EmployeeCount','EmployeeNumber','JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus',  'MonthlyRate', 'Over18', 'OverTime', 'PerformanceRating','StandardHours', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'],axis=1) # 필요없는 feature 삭제
    df2 = df1.loc[:,['Age','DailyRate','DistanceFromHome','EnvironmentSatisfaction','Gender','HourlyRate', 'JobSatisfaction','MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel','WorkLifeBalance','Attrition']] # 종속변수값 재배치
    # df2.info() # 전처리 해야할 object 타입 확인
    X = df2.iloc[:,:-1].values # 독립변수 값들 OneHot인코딩을 위해 나누기
    cy = df2.iloc[:,-1:].values
    y = np.array([1 if x[0] == "Yes" else 0 for x in cy]) # 리스트 컴프리핸션으로 종속변수 YES값은 1로 NO값은 0으로 만들어 NUMPY배열로 변환환
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [4])], remainder='passthrough')
    X = ct.fit_transform(X)
    return X,y

def random_forest_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    with st.echo(code_location="below"):
        model_path = "Data/pkl/RandomForest_model.pkl"
        model = joblib.load(model_path)
        st.write("## Randomforest")
    
    train_pred_dt = model.predict(X_train) 
    test_pred_dt = model.predict(X_test)
    
    predict_button_dt1 = st.button('예측')

    if predict_button_dt1:        
        st.write(f'Train-set : {model.score(X_train, y_train)}')
        st.write(f'Test-set : {model.score(X_test, y_test)}')

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred_dt)
    st.write(accuracy)

    st.header("random_forest")
    # 첫번째 행
    r1_col1, r1_col2, r1_col3, r1_col4  = st.columns(4)
    나이 = r1_col1.slider("나이",20,70)
    일일급여 = r1_col2.slider("일일급여", 1, 1500)
    회사와의거리 = r1_col3.slider("회사와의거리", 1, 30)
    근무환경만족 = r1_col4.slider("근무환경만족", 1, 4)

    # 두번째 행
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    시간당임금 = r2_col1.slider("시간당임금",15, 100)
    직업만족도 = r2_col2.slider("직업만족도",1,4)
    월수입 = r2_col3.slider('월수입',1,4)
    이직회사수 = r2_col4.slider('이직회사수',0,9)

    # 세번째 행
    r3_col1, r3_col2, r3_col3, r3_col4 = st.columns(4)
    급여인상비율 = r3_col1.slider("급여인상률",10,25)
    동료관계만족도 = r3_col2.slider('동료관계만족도',1,4)
    스톡옵션레벨 = r3_col3.slider('스톡옵션레벨',0,3)
    워라벨 = r3_col4.slider('워라벨',1,4)

    성별 = st.selectbox(
        '성별',
    ('남자', '여자'))

    predict_button = st.button("퇴사유무 예측")
    
    if predict_button:
            variable1 = np.array([나이, 일일급여, 회사와의거리, 근무환경만족, 성별=="남자", 시간당임금, 직업만족도, 월수입, 이직회사수, 급여인상비율, 동료관계만족도, 스톡옵션레벨, 워라벨])
            model1 = joblib.load('Data/pkl/RandomForest_model.pkl')
            pred1 = model1.predict([variable1])
            if pred1 == 1:
                st.write("퇴사")
            else:
                st.write("근속")

def lightGBM_model(df):
    X,y = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    with st.echo(code_location="below"):
        model_path = "Data/pkl/LightGBM_model.pkl"
        model = joblib.load(model_path)
        st.write("## lightGBM_model")
    
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
        model_path = "Data/pkl/XGBoost.pkl"
        xgb_model = joblib.load(model_path)
        st.write("## XGBoost_model")

    X,y = data_preprocessing(df)
    # # 훈련 및 검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # # train_pred_dt = xgb.predict(X_train) 
    # # test_pred_dt = xgb.predict(X_valid)
    y_pred = xgb_model.predict(X_valid)
    # # 정확도 계산

    predict_button_dt3 = st.button('예측!')

    if predict_button_dt3:        
        st.write(f'Train-set : {xgb_model.score(X_train, y_train)}')
        st.write(f'Test-set : {xgb_model.score(X_valid, y_valid)}')

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_valid, y_pred)
    st.write(accuracy)