import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train) # 훈련 세트로 학습


def data_processing():
    # csv데이터 호출
    df = pd.read_csv('Data/test_scores.csv')

    # 불 필요한 열 데이터 제거
    df1 = df.drop(['school','classroom','student_id'], axis=1)

    # Seaborn 차트 생성 및 확인 용도
    draw_chart(df1)

    # 데이터에 필요한 훈련셋, 테스트셋 분류
    X = df1.iloc[:, :-1].values
    y = df1.iloc[:, -1].values

    data_set = sort_data_to_train(hot_encoding(X), y)
    return data_set

def hot_encoding(X):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0, 1, 2, 4, 5])], remainder='passthrough')
    X = ct.fit_transform(X)
    return X

def sort_data_to_train(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    data_list = [X_train, X_test, y_train, y_test]
    return data_list

        

def draw_barchart(df):
    sns.barplot(data=df1, x='gender', y='posttest')

def create_model(processed_data):
    X_train = processed_data[0]
    y_train = processed_data[1]
    X_test = processed_data[2]
    y_train = processed_data[3]

    reg = LinearRegression()
    reg.fit(X_train, y_train) # 훈련 세트로 학습
    
    compare_model(X_test)


def compare_model():
    y_pred = reg.predict(X_test)
    st.write(y_pred)


processed_data = data_processing()
reg = create_model(processed_data)
