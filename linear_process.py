import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pre_process


def describe_linear_model(df):
    st.write(df.describe(include='all'))

def linear_processed_df(df, s_df, comparison1):
    df1 = df.drop(['school','classroom','student_id'], axis=1)

    X = df1.iloc[:, :-1].values
    y = df1.iloc[:, -1].values

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0, 1, 2, 4, 5])], remainder='passthrough')
    X = ct.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    reg = LinearRegression()
    reg.fit(X_train, y_train) # 훈련 세트로 학습
    reg.coef_
    reg.intercept_
    y_pred = reg.predict(X_test)

    mean_absolute_error(y_test, y_pred) # 실제 값, 예측 값 # MAE
    mean_squared_error(y_test, y_pred) # MSE
    mean_squared_error(y_test, y_pred, squared=False) # RMSE
    r2_score(y_test, y_pred) # R2


    st.write("## 컬럼별 상관 관계")
    fig = px.imshow(df1.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    fig = px.imshow(s_df.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)
    st.write('---')



    st.write("## 파이차트 ")


    a_labels = df1['school_setting'].unique()
    a_values = df1['school_setting'].value_counts()
    st_labels = df1['school_type'].unique()
    st_values = df1['school_type'].value_counts()
    gender_labels = df1['school_type'].unique()
    gender_values = df1['school_type'].value_counts()
    teaching_labels = df1['teaching_method'].unique()
    teaching_values = df1['teaching_method'].value_counts()

    
    specs = [[{'type': 'pie'}, {'type': 'pie'}],
         [{'type': 'pie'}, {'type': 'pie'}]]

    fig = make_subplots(rows=2, cols=2, specs=specs)

    # subplot에 그래프 추가
    fig.add_trace(go.Pie(labels=a_labels, values=a_values, pull=[0.1, 0, 0], textinfo='label+percent', insidetextorientation='radial'), row=1, col=1)
    fig.add_trace(go.Pie(labels=st_labels, values=st_values, pull=[0.1, 0], textinfo='label+percent', insidetextorientation='radial'), row=1, col=2)
    fig.add_trace(go.Pie(labels=gender_labels, values=gender_values, pull=[0.1, 0], textinfo='label+percent', insidetextorientation='radial'), row=2, col=1)
    fig.add_trace(go.Pie(labels=teaching_labels, values=teaching_values, pull=[0.1, 0], textinfo='label+percent', insidetextorientation='radial'), row=2, col=2)

    # subplot 레이아웃 설정
    fig.update_layout(height=600, width=800, title_text="OjectType의 데이터 분포")

    # subplot을 streamlit에 출력
    st.plotly_chart(fig)



    # 테이블로 평가

    st.write("## LinearRegression 산점도")
    comparison = pd.DataFrame(
        {
            '실제값' : y_test, # 실제값
            '예측값' : y_pred, #  머신러닝 모델을 통해 예측한 예측값
        }
    )
    
    fig = make_subplots(rows=1, cols=2)

    colors = ['red', 'blue']
    # fig.add_trace(go.Scatter(comparison, x="실제값", y="예측값", color_discrete_sequence=colors),row=1, col=1)
    # fig.add_trace(go.Scatter(comparison1, x="실제값", y="예측값", color_discrete_sequence=colors),row=1, col=2)
    # fig = px.scatter(comparison, x="실제값", y="예측값", color_discrete_sequence=colors)
    # st.plotly_chart(fig)
    # fig = px.scatter(comparison1, x="실제값", y="예측값", color_discrete_sequence=colors)
    st.plotly_chart(fig)


    scatter_trace = go.Scatter(x=comparison["실제값"], y=comparison["예측값"], mode="markers", 
                               marker=dict(color=colors, size=8))
    fig.add_trace(scatter_trace, row=1, col=1)