import streamlit as st
import seaborn as sns
import pandas as pd
import joblib
from PIL import Image

# 초기 배경 이미지
image = Image.open('image/m_img.png')
st.image(image, width=600)

def show_first_ml():
    data_url = "Data/test_scores.csv"
    df = pd.read_csv(data_url) # URL로 CSV 불러오기
    df1 = df.drop(['school','classroom','student_id'], axis=1)

    image = Image.open('image/image.png')

    st.image(image, width=600)


    st.write("# 모델 통해 예측해 보기")

    with st.echo(code_location="below"):
        model_path = "Data/multi_LinearRegression_model.pkl"
        model = joblib.load(model_path)
        st.write("## 다중 선형 회귀 모델")

    st.write("---")

    with st.echo(code_location="below"):
        # 학교 지역 (라디오 버튼)
        area = st.radio(
            label="지역", # 상단 표시되는 이름
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
        students = st.slider(
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

    with st.echo(code_location="below"):
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


    if st.button('전처리 전 데이터'):
        st.write(df) # 자동으로 표 그려줌

    if st.button('전처리 후 데이터'):
        st.write(df1) # 자동으로 표 그려줌

    # Using object notation
add_selectbox = st.sidebar.selectbox(
    "어떤 머신러닝을 보시겠어요?",
    ("학생 점수 예측", "추가예정", "추가예정")
)
show_button = st.sidebar.button(
            label="보기", # 버튼 내부 표시되는 이름
        )
if add_selectbox == "학생 점수 예측":
    if show_button:
        show_first_ml()


