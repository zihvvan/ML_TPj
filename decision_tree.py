import numpy as np
import pandas as pd
import graphviz
import multiprocessing
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score, confusion_matrix #accuracy_score(y_test, y_pred)

def decision_tree_preprocessing(df):
    df1 = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
    df2 = df1.loc[:,['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked','OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager','JobRole', 'Attrition']]
    X = df2.iloc[:,:-1].values
    y = df2.iloc[:,-1:].values
    df2['Attrition'] = df2['Attrition'].map({'No':0, 'Yes': 1})
    df_dummy = pd.get_dummies(df2, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime', 'JobRole'] , drop_first = True)
    X = df_dummy
    df2_dummy = pd.get_dummies(df2['Attrition'])
    y = df2_dummy

    X_train, X_test, y_train, y_test = train_test_split(X.drop(['Attrition'], axis=1), df2['Attrition'], test_size=0.2, random_state=10)

    model = DecisionTreeClassifier(random_state = 10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(acc)