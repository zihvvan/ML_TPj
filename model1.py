import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image
from math import sqrt
from sklearn.metrics import mean_absolute_error
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

    fig = px.imshow(concated_df.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
    fig.update_layout(title='컬럼별 상관관계',xaxis_nticks=36)
    fig.layout.update({'width':800, 'height':600})
    st.plotly_chart(fig)
    st.write('---')
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
    st.write(f"이 모델의 최적의 Alpha 값 :  {best_params['alpha']}")
    st.write(f"이 모델의 최적의 Max_iter 횟수  :  {best_params['max_iter']}")
    # 관계도
    st.write(model.coef_)
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
            '실제값' : y_test, # 실제값
            '예측값' : y_pred, #  머신러닝 모델을 통해 예측한 예측값
        }
    )
    comparison
    colors = ['red', 'blue']
    import plotly.express as px
    fig = px.scatter(comparison, x="실제값", y="예측값")
    st.plotly_chart(fig)

def poly_model(df):
    df2 = pre_process.pre_processing(df)
    scaled_df = scaler_df(df2)
    pre_processed_df = make_polynomial_df(scaled_df)
    X, y = split_dataset(pre_processed_df)
    run_model(X, y)

# 이미지 불러오기
def linear_model():

    with st.echo(code_location="below"):
        model_path = "Data/pkl/multi_LinearRegression_model.pkl"
        model = joblib.load(model_path)
        st.write("## 다중 선형 회귀 모델")
        st.write("모델링 소요시간이 짧으며 구현과 해석이 쉬운 선형 회귀 모델을 사용하였습니다.")

    st.write("---")
    with st.echo(code_location="below"):
        # 학교 지역 (라디오 버튼)
        st.write("**학생들이 다니는 학교의 지역을 선택할 수 있습니다.**")
        area = st.radio(
            label="지역", # 상단 표시되는 이름
            options=["Urban", "Suburban","Rural"], # 선택 옵션
        )
        
    st.write("---")
    with st.echo(code_location="below"):
        # 학교 종류 (라디오 버튼)
        st.write("**학생들이 다니는 학교의 종류를 구분하여 선택할 수 있습니다.**")
        school_type = st.radio(
            label="학교 타입", # 상단 표시되는 이름
            options=["국립", "사립"], # 선택 옵션
        )

    st.write("---")    
    with st.echo(code_location="below"):
        # 수업 방식 (라디오 버튼)
        st.write("**학생들이 듣는 수업 방식을 선택할 수 있습니다.**")
        teaching_method = st.radio(
            label="수업 타입", # 상단 표시되는 이름
            options=["일반", "체험"], # 선택 옵션
        )

    st.write("---")    
    with st.echo(code_location="below"):
        # 반 학생수 (숫자)
        st.write("**학급의 인원을 선택할 수 있습니다.(10~30명)**")
        students = st.slider(
            label="학급 인원", # 상단 표시되는 이름
            min_value=10.0, # 최솟값
            max_value=30.0, # 최댓값
            step=1.0, # 입력 단위
        )

    st.write("---")    
    with st.echo(code_location="below"):
        # 성별 입력 (라디오 버튼)
        st.write("**학생의 성별을 선택할 수 있습니다.**")
        gender = st.radio(
            label="성별", # 상단 표시되는 이름
            options=["Male", "Female"], # 선택 옵션
        )

    st.write("---")
    with st.echo(code_location="below"):
        # 점심 유무
        st.write("**시험 전 학생들의 점심식사 유무를 선택할 수 있습니다.**")
        lunch = st.radio(
            label="점심식사 유무", # 상단 표시되는 이름
            options=["먹음", "안먹음"], # 선택 옵션
        )

    st.write("---")
    with st.echo(code_location="below"):
        # 사전 시험 (숫자)
        st.write("**학생들의 사전 시험점수를 선택할 수 있습니다.(0~100점)**")
        pretest = st.number_input(
            label="사전 시험", # 상단 표시되는 이름
            min_value=0.0, # 최솟값
            max_value=100.0, # 최댓값
            step=1.0, # 입력 단위
            value=50.0 # 기본값
        )
    input_data_set ={
                            "area": [area], 
                            "school_type": [school_type], 
                            "teaching_method": [teaching_method], 
                            "students": [students], 
                            "gender": [gender], 
                            "lunch": [lunch], 
                            "pretest": [pretest]
                        }

    df_input_data_set = pd.DataFrame(input_data_set)

    st.write("---")

    with st.echo(code_location="below"):
        st.write("**예측하기**")
        # 실행 버튼
        play_button = st.button(
            label="예측하기", # 버튼 내부 표시되는 이름
        )


    with st.echo(code_location="below"):
        # 실행 버튼이 눌리면 모델을 불러와서 예측한다
        if play_button:
            input_data_set ={
                            "area": [area], 
                            "school_type": [school_type], 
                            "teaching_method": [teaching_method], 
                            "students": [students], 
                            "gender": [gender], 
                            "lunch": [lunch], 
                            "pretest": [pretest]
                        }
            input_values = [[area == "Urban",area =="Suburban",school_type == "국립",teaching_method == "일반",students,gender=="Male",lunch=="안먹음",pretest]]
            pred = model.predict(input_values)
            pred_df = pd.DataFrame(pred)
            st.markdown(f"<div style='text-align:center; font-size:24px'>예측 학생 점수 :{(pred_df.iloc[0,0]).round(1)}</div>", unsafe_allow_html=True)
            
            st.write()