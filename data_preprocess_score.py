import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import pre_process

def show_processed_df(df):
    st.write(df)

def scaler_df(df2):
    # 스케일링
    scaler = preprocessing.MinMaxScaler() # 최대최소값을 이용한 스케일러 
    scaled_data = scaler.fit_transform(df2.loc[:,['학생수','사전점수','시험점수']])
    features = df2.loc[:,['Suburban','Urban', 'Public','Standard', 'Male','free lunch']]
    scaled_data1 = pd.DataFrame(scaled_data,columns=['학생수','사전점수','시험점수'])
    concated_df = pd.concat([scaled_data1,features],axis=1)
    return concated_df

def make_polynomial_df(concated_df):
    # 다항회귀 추가 (복잡도를 높이기 위해 추가)
    poly_data = concated_df.values
    poly_columns = concated_df.columns
    polynomial_transformer = PolynomialFeatures(2) # 2차원으로 다항회귀 
    polynomial_data = polynomial_transformer.fit_transform(poly_data)
    polynomial_features_names = polynomial_transformer.get_feature_names_out(poly_columns)
    poly_df = pd.DataFrame(polynomial_data, columns=polynomial_features_names).drop('1', axis=1)
    st.write(poly_df.describe(include='all'))

    # fig = px.imshow(concated_df.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    # fig.update_layout(title='컬럼별 상관관계',xaxis_nticks=36)
    # fig.layout.update({'width':800, 'height':600})
    # st.plotly_chart(fig)
    # st.write('---')
    return poly_df

def split_dataset(pre_processed_df):
    # 테스트 셋 나누기 작업
    X = pre_processed_df.drop('시험점수',axis=1)
    y = pre_processed_df['시험점수']

    return X, y

def run_model(X, y):
    # 테스트셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # 모델 선언(선형, 라쏘모델, 그리드 서치)
    hyper_params = {
        'alpha' : [0.01, 0.1, 1, 10],
        'max_iter' : [100, 500, 1000, 2000, 3000]
    }
    model = LinearRegression() # 선형회귀 
    model.fit(X_train, y_train) # 훈련 세트로 학습

    y_pred = model.predict(X_test)

    lasso_model = Lasso()
    hyper_param_tuner = GridSearchCV(lasso_model,hyper_params,cv=5)
    hyper_param_tuner.fit(X,y)
    best_params = hyper_param_tuner.best_params_

    # # 관계도
    coef = model.coef_
    intercept = model.intercept_


    # 성능평가
    
    train_score = model.score(X_train, y_train) # 훈련 세트
    test_score = model.score(X_test, y_test) # 테스트 세트

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_test_predict) # 실제 값, 예측 값 # MAE
    # mse = mean_squared_error(y_test, y_test_predict) # MSE
    rmse = mean_squared_error(y_test, y_test_predict, squared=False) # RMSE

    r2 = r2_score(y_test, y_test_predict) # R2

    index = ["다항선형회귀모델(Lasso)"]
    total_info = {"Intercept": intercept, "MAE" : mae, "RMSE": rmse, "R2" : r2, "그리드 alpha" : best_params['alpha'], "그리드 max_iter": best_params['max_iter']}
    total_df = pd.DataFrame([total_info], index=index)
    # st.write(total_df)
    # 테이블로 평가
    comparison = pd.DataFrame(
        {
            '실제값' : y_test, # 실제값
            '예측값' : y_pred, #  머신러닝 모델을 통해 예측한 예측값
        }
    )
    return comparison, total_df

def poly_model(df):
    df2 = pre_process.pre_processing(df)
    scaled_df = scaler_df(df2)
    pre_processed_df = make_polynomial_df(scaled_df)
    X, y = split_dataset(pre_processed_df)
    comparison, total_df = run_model(X, y)
    # draw_table(total_df1, total_df2)
    return scaled_df, comparison, total_df

def linear_process(df):
    df1 = df.drop(['school','classroom','student_id'], axis=1)

    X = df1.iloc[:, :-1].values
    y = df1.iloc[:, -1].values

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0, 1, 2, 4, 5])], remainder='passthrough')
    X = ct.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    reg = LinearRegression()
    reg.fit(X_train, y_train) # 훈련 세트로 학습
    intercept = reg.intercept_
    y_pred = reg.predict(X_test)

    train_score = reg.score(X_train, y_train) # 훈련 세트
    test_score = reg.score(X_test, y_test) # 테스트 세트

    mae = mean_absolute_error(y_test, y_pred) # 실제 값, 예측 값 # MAE
    # mse = mean_squared_error(y_test, y_pred) # MSE
    rmse = mean_squared_error(y_test, y_pred, squared=False) # RMSE
    r2 = r2_score(y_test, y_pred) # R2

    index = ["다중선형회귀모델"]
    total_info = {"Intercept": intercept, "MAE" : mae,"RMSE": rmse, "R2" : r2, "그리드 alpha" : "X", "그리드 max_iter": "X"}
    total_df = pd.DataFrame([total_info], index=index)

    return total_df

def draw_table(total_df1, total_df2):
    st.header("지표 분석")
    total_set = pd.concat([total_df2, total_df1], axis=0, join='inner')
    st.write(total_set)
