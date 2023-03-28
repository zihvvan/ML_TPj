import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def pre_processing(df):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)


    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test) #


    # 정확도를 계산하여 모델의 성능을 평가합니다.
    accuracy = accuracy_score(y_test, test_pred)
    st.write(accuracy)