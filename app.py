import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
from PIL import Image
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge

image1 = Image.open('image/m_img.png')
st.image(image1, width=600)
data_url = "Data/test_scores.csv"
df = pd.read_csv(data_url) # URL로 CSV 불러오기
df1 = df.drop(['school','classroom','student_id'], axis=1)
st.write("전처리 전 데이터") # 마크다운으로 꾸미기
st.write(df1)
st.write("전처리 후 데이터") # 마크다운으로 꾸미기
df1 = pd.get_dummies(df1, columns=['school_setting','school_type','teaching_method','gender','lunch'], drop_first=True)
st.write(df1)

# 스케일링
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(df1.loc[:,['n_student','pretest','posttest']])
features = df1.loc[:,['school_setting_Suburban','school_setting_Urban', 'school_type_Public','teaching_method_Standard', 'gender_Male','lunch_Qualifies for reduced/free lunch']]
scaled_data1 = pd.DataFrame(scaled_data,columns=['n_student','pretest','posttest'])
concated_df = pd.concat([scaled_data1,features],axis=1)

# 다항회귀 추가
poly_data = concated_df.values
poly_columns = concated_df.columns
polynomial_transformer = PolynomialFeatures(2)
polynomial_data = polynomial_transformer.fit_transform(poly_data)
polynomial_features_names = polynomial_transformer.get_feature_names_out(poly_columns)
poly_df = pd.DataFrame(polynomial_data, columns=polynomial_features_names).drop('1', axis=1)
X = poly_df.drop('posttest',axis=1)
y = poly_df['posttest']

# 테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# 모델 선언(선형회귀)
model = LinearRegression()
# model = Lasso(alpha=0.001, max_iter=1000)
# model = Ridge(alpha=0.001, max_iter=1000)
model.fit(X_train, y_train) # 훈련 세트로 학습

# 예측
y_pred = model.predict(X_test)

# 관계도
st.write("관계")
st.write(model.coef_)
st.write("갭차이")
st.write(model.intercept_)

# 성능평가
st.write("훈련셋 Score")
st.write(model.score(X_train, y_train)) # 훈련 세트
st.write("테스트 셋 Score")
st.write(model.score(X_test, y_test)) # 테스트 세트

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

mse = mean_absolute_error(y_train, y_train_predict)
st.write("train-set에서 성능")
st.write(sqrt(mse))

mse = mean_absolute_error(y_test, y_test_predict)
st.write("test-set에서 성능")
st.write(sqrt(mse))

# 테이블로 평가
comparison = pd.DataFrame(
    {
        'actual' : y_test, # 실제값
        'pred': y_pred, #  머신러닝 모델을 통해 예측한 예측값
    }
)

import plotly.express as px
fig = px.scatter(comparison, x="pred", y="actual")
fig.show()





# def show_first_ml():

#     st.write("# 모델 통해 예측해 보기")

#     with st.echo(code_location="below"):
#         model_path = "Data/LinearRegression.pkl"
#         model = joblib.load(model_path)
#         st.write("## 다중 선형 회귀 모델")

#     st.write("---")

#     with st.echo(code_location="below"):
#         # 학교 지역 (라디오 버튼)
#         area = st.radio(
#             label="지역", # 상단 표시되는 이름
#             options=["Urban", "Suburban","Rural"], # 선택 옵션
#         )
            

#     with st.echo(code_location="below"):
#         # 학교 종류 (라디오 버튼)
#         school_type = st.radio(
#             label="학교 타입", # 상단 표시되는 이름
#             options=["국립", "사립"], # 선택 옵션
#         )

#     with st.echo(code_location="below"):
#         # 수업 방식 (라디오 버튼)
#         teaching_method = st.radio(
#             label="수업 타입", # 상단 표시되는 이름
#             options=["일반", "체험"], # 선택 옵션
#         )

#     with st.echo(code_location="below"):
#         # 반 학생수 (숫자)
#         students = st.slider(
#             label="학급 인원", # 상단 표시되는 이름
#             min_value=10.0, # 최솟값
#             max_value=30.0, # 최댓값
#             step=1.0, # 입력 단위
#         )

#     with st.echo(code_location="below"):
#         # 성별 입력 (라디오 버튼)
#         gender = st.radio(
#             label="성별", # 상단 표시되는 이름
#             options=["Male", "Female"], # 선택 옵션
#         )


#     with st.echo(code_location="below"):
#         # 점심 유무
#         lunch = st.radio(
#             label="점심식사 유무", # 상단 표시되는 이름
#             options=["먹음", "안먹음"], # 선택 옵션
#         )


#     with st.echo(code_location="below"):
#         # 사전 시험 (숫자)
#         pretest = st.number_input(
#             label="사전 시험", # 상단 표시되는 이름
#             min_value=0.0, # 최솟값
#             max_value=100.0, # 최댓값
#             step=1.0, # 입력 단위
#             value=50.0 # 기본값
#         )
#     input_data_set ={
#                             "area": [area], 
#                             "school_type": [school_type], 
#                             "teaching_method": [teaching_method], 
#                             "students": [students], 
#                             "gender": [gender], 
#                             "lunch": [lunch], 
#                             "pretest": [pretest]
#                         }

#     df_input_data_set = pd.DataFrame(input_data_set)

#     with st.echo(code_location="below"):
#         # 실행 버튼
#         play_button = st.button(
#             label="예측하기", # 버튼 내부 표시되는 이름
#         )


#     with st.echo(code_location="below"):
#         # 실행 버튼이 눌리면 모델을 불러와서 예측한다
#         if play_button:
#             input_data_set ={
#                             "area": [area], 
#                             "school_type": [school_type], 
#                             "teaching_method": [teaching_method], 
#                             "students": [students], 
#                             "gender": [gender], 
#                             "lunch": [lunch], 
#                             "pretest": [pretest]
#                         }
#             input_values = [[area == "Urban",area =="Suburban",school_type == "국립",teaching_method == "일반",students,gender=="Male",lunch=="안먹음",pretest]]
#             pred = model.predict(input_values)
#             pred_df = pd.DataFrame(pred)
#             st.markdown(f"<div style='text-align:center; font-size:24px'>예측 학생 점수 :{(pred_df.iloc[0,0]).round(1)}</div>", unsafe_allow_html=True)
#     if st.button('전처리 전 데이터'):
#         st.write(df) # 자동으로 표 그려줌
#     if st.button('전처리 후 데이터'):
#         st.write(df1) # 자동으로 표 그려줌
# show_first_ml()