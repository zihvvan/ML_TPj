import numpy as np
import streamlit as st
import pandas as pd
import graphviz
import plotly.express as px
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import export_graphviz
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

    # 예측 결과를 바탕으로 confusion matrix 생성
    cf_matrix = confusion_matrix(y_test, y_pred)

    # 그룹 이름과 개수를 통해 label 생성
    group_names = ['True Negative','False Positive','False Negative','True Positive']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    heatm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    fig = heatm.get_figure()
    # seaborn을 사용한 heatmap 시각화
    st.pyplot(fig)

    fig = plt.figure(figsize=(30, 15))
    plot_tree(model, max_depth=3, fontsize=10, feature_names=df_dummy.columns) # 독립변수명을 추가로 지정

    # Matplotlib 그림을 Streamlit에서 출력
    st.pyplot(fig)
