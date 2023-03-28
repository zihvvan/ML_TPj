import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
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

def linear_processed_df(df):
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

    fig = px.imshow(df1.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    fig.update_layout(title='컬럼별 상관관계',xaxis_nticks=36)
    fig.layout.update({'width':800, 'height':600})
    st.plotly_chart(fig)
    st.write('---')

    # 테이블로 평가
    comparison = pd.DataFrame(
        {
            '실제값' : y_test, # 실제값
            '예측값' : y_pred, #  머신러닝 모델을 통해 예측한 예측값
        }
    )
    comparison
    colors = ['red', 'blue']
    import plotly.express as px
    fig = px.scatter(comparison, x="실제값", y="예측값")
    st.plotly_chart(fig)