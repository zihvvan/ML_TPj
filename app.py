import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression


data_url = "Data/test_scores.csv"
df = pd.read_csv(data_url) # URL로 CSV 불러오기
df1 = df.drop(['school','classroom','student_id'], axis=1)


st.write(df1) # 자동으로 표 그려줌

st.write("# 모델 통해 예측해 보기")

with st.echo(code_location="below"):
    model_path = "Data/multi_LinearRegression_model.pkl"
    model = joblib.load(model_path)
    st.write("## 다중 선형 회귀 모델")

st.write("---")

with st.echo(code_location="below"):
    # 학교 지역 (라디오 버튼)
    area = st.radio(
        label="지", # 상단 표시되는 이름
        options=["Urban", "Suburban","Rural"], # 선택 옵션
        # index=0 # 기본 선택 인덱스
        # horizontal=True # 가로 표시 여부
    )
        

with st.echo(code_location="below"):
    # 학교 종류 (라디오 버튼)
    school_type = st.radio(
        label="학교 타입", # 상단 표시되는 이름
        options=["국립", "사립"], # 선택 옵션
        # index=0 # 기본 선택 인덱스
        # horizontal=True # 가로 표시 여부
    )

with st.echo(code_location="below"):
    # 수업 방식 (라디오 버튼)
    teaching_method = st.radio(
        label="수업 타입", # 상단 표시되는 이름
        options=["일반", "체험"], # 선택 옵션
        # index=0 # 기본 선택 인덱스
        # horizontal=True # 가로 표시 여부
    )

with st.echo(code_location="below"):
    # 반 학생수 (숫자)
    students = st.number_input(
        label="학급 인원", # 상단 표시되는 이름
        min_value=10.0, # 최솟값
        max_value=30.0, # 최댓값
        step=1.0, # 입력 단위
        # value=25.0 # 기본값
    )

with st.echo(code_location="below"):
    # 성별 입력 (라디오 버튼)
    gender = st.radio(
        label="성별", # 상단 표시되는 이름
        options=["Male", "Female"], # 선택 옵션
        # index=0 # 기본 선택 인덱스
        # horizontal=True # 가로 표시 여부
    )


with st.echo(code_location="below"):
    # 점심 유무
    lunch = st.radio(
        label="점심식사 유무", # 상단 표시되는 이름
        options=["먹음", "안먹음"], # 선택 옵션
    )


with st.echo(code_location="below"):
    # 사전 시험 (숫자)
    pretest = st.number_input(
        label="사전 시험", # 상단 표시되는 이름
        min_value=0.0, # 최솟값
        max_value=100.0, # 최댓값
        step=1.0, # 입력 단위
        # value=25.0 # 기본값
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
st.write(df_input_data_set)

with st.echo(code_location="below"):
    # 실행 버튼
    play_button = st.button(
        label="데이터생성", # 버튼 내부 표시되는 이름
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

    input_values = [[area == "Urban",area =="Suburban",school_type == "국립",teaching_method == "일반",students,gender=="Male",lunch=="먹음",pretest]]
    pred = model.predict(input_values)
    pred_df = pd.DataFrame(pred)
    st.write(pred_df.iloc[0,0])
    st.markdown(f"<div style='text-align:center; font-size:24px'>예측 점수 :{pred_df}</div>", unsafe_allow_html=True)











# def data_processing():
#     # csv데이터 호출
#     df = pd.read_csv('Data/test_scores.csv')

#     # 불 필요한 열 데이터 제거
#     df1 = df.drop(['school','classroom','student_id'], axis=1)

#     # Seaborn 차트 생성 및 확인 용도
#     draw_chart(df1)

#     # 데이터에 필요한 훈련셋, 테스트셋 분류
#     X = df1.iloc[:, :-1].values
#     y = df1.iloc[:, -1].values

#     data_set = sort_data_to_train(hot_encoding(X), y)
#     return data_set

# def hot_encoding(X):
#     ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0, 1, 2, 4, 5])], remainder='passthrough')
#     X = ct.fit_transform(X)
#     return X

# def sort_data_to_train(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#     data_list = [X_train, X_test, y_train, y_test]
#     return data_list

# def draw_barchart(df):
#     sns.barplot(data=df1, x='gender', y='posttest')

# def create_model(processed_data):
#     X_train = processed_data[0]
#     y_train = processed_data[1]
#     X_test = processed_data[2]
#     y_train = processed_data[3]

#     reg = LinearRegression()
#     reg.fit(X_train, y_train) # 훈련 세트로 학습
    
#     compare_model(X_test)

# def compare_model():
#     y_pred = reg.predict(X_test)
#     st.write(y_pred)


# processed_data = data_processing()
# reg = create_model(processed_data)


