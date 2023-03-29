import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from xgboost import XGBClassifier
from PIL import Image
from math import sqrt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def data_preprocessing(df):
    df1 = df.drop(['BusinessTravel','Department','Education','EducationField','EmployeeCount','EmployeeNumber','JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus',  'MonthlyRate', 'Over18', 'OverTime', 'PerformanceRating','StandardHours', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'],axis=1) # 필요없는 feature 삭제
    df2 = df1.loc[:,['Age','DailyRate','DistanceFromHome','EnvironmentSatisfaction','Gender','HourlyRate', 'JobSatisfaction','MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel','WorkLifeBalance','Attrition']] # 종속변수값 재배치
    # df2.info() # 전처리 해야할 object 타입 확인
    X = df2.iloc[:,:-1].values # 독립변수 값들 OneHot인코딩을 위해 나누기
    cy = df2.iloc[:,-1:].values
    y = np.array([1 if x[0] == "Yes" else 0 for x in cy]) # 리스트 컴프리핸션으로 종속변수 YES값은 1로 NO값은 0으로 만들어 NUMPY배열로 변환환
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [4])], remainder='passthrough')
    X = ct.fit_transform(X)
    return X,y

def linear_model():

    with st.echo(code_location="below"):
        model_path = "Data/pkl/multi_LinearRegression_model.pkl"
        model = joblib.load(model_path)
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

def random_forest_model(df):
    # X,y = data_preprocessing(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    # with st.echo(code_location="below"):
    #     model_path = "Data/pkl/RandomForest_model.pkl"
    #     model = joblib.load(model_path)
    #     st.write("## Randomforest")
    #     st.write("**의사 결정 트리(Decision Tree)를 기반으로 하는 앙상블(Ensemble) 학습 알고리즘**")
    #     st.write("여러 개의 의사 결정 트리 생성 및 각각의 트리가 예측한 결과를 투표(voting)통해 최종 결과를 도출할 수 있으며, 일반화(generalization) 성능을 향상시키고, 과적합(overfitting)을 방지할 수 있습니다.")
        
    
    # train_pred_dt = model.predict(X_train) 
    # test_pred_dt = model.predict(X_test)
    
    # predict_button_dt1 = st.button('예측')

    # if predict_button_dt1:        
    #     st.write(f'Train-set : {model.score(X_train, y_train)}')
    #     st.write(f'Test-set : {model.score(X_test, y_test)}')

    # # 정확도를 계산하여 모델의 성능을 평가합니다.
    # accuracy = accuracy_score(y_test, test_pred_dt)
    # st.write(accuracy)

    r1_col1, r1_col2, r1_col3, r1_col4  = st.columns(4)
    나이 = r1_col1.slider("나이",20,70,key="test1")
    일일급여 = r1_col2.slider("일일급여", 110, 1500,key="test2")
    회사와의거리 = r1_col3.slider("회사와의거리", 1, 30,key="test3")
    근무환경만족 = r1_col4.slider("근무환경만족", 1, 4,key="test4")

    # 두번째 행
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    시간당임금 = r2_col1.slider("시간당임금",30, 100,key="test5")
    직업만족도 = r2_col2.slider("직업만족도",1,4,key="test6")
    월수입 = r2_col3.slider('월수입',1000,20000,key="test7")
    이직회사수 = r2_col4.slider('이직회사수',0,9,key="test8")

    # 세번째 행
    r3_col1, r3_col2, r3_col3, r3_col4 = st.columns(4)
    급여인상비율 = r3_col1.slider("급여인상률",10,25,key="test9")
    동료관계만족도 = r3_col2.slider('동료관계만족도',1,4,key="test10")
    스톡옵션레벨 = r3_col3.slider('스톡옵션레벨',0,3,key="test11")
    워라벨 = r3_col4.slider('워라벨',1,4,key="test12")

    성별 = st.selectbox(
        '성별',
    ('남자', '여자'),key="test13")

    predict_button = st.button("퇴사유무 예측",key="test14")
    
    if predict_button:
            variable1 = np.array([나이, 일일급여, 회사와의거리, 근무환경만족, 성별=="남자", 시간당임금, 직업만족도, 월수입, 이직회사수, 급여인상비율, 동료관계만족도, 스톡옵션레벨, 워라벨])
            model1 = joblib.load('Data/pkl/RandomForest_model.pkl')
            pred1 = model1.predict([variable1])
            # st.write(pred1)
            if pred1 == 1:
                st.write("퇴사")
            else:
                st.write("근속")

def lightGBM_model(df):
    # X,y = data_preprocessing(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

    # with st.echo(code_location="below"):
    #     model_path = "Data/pkl/LightGBM_model.pkl"
    #     # model_path = "Data/pkl/Gridedlightgbm.pkl" #그리드 pkl파일
    #     model = joblib.load(model_path)
    #     st.write("## lightGBM_model")
    #     st.write("**Gradient Boosting 알고리즘 중에서도 분할(split) 방법을 최적화하여 학습하는 알고리즘입니다.**")
    #     st.write("매우 빠른 학습 속도와 높은 정확도를 동시에 달성할 수 있습니다.")
    #     st.write("또한, leaf-wise(잎노드 방식) 트리 성장 방법을 사용하여 깊은 트리를 만들 수 있습니다.")
    #     st.write("regularization 및 early stopping 기능을 제공하여 과적합을 방지할 수 있습니다.")

    
    # train_pred_dt = model.predict(X_train) 
    # test_pred_dt = model.predict(X_test)
    
    # predict_button_dt2 = st.button('예측하기')

    # if predict_button_dt2:        
    #     # st.write(f'Train-set : {model.score(X_train, y_train)}')
    #     # st.write(f'Test-set : {model.score(X_test, y_test)}')
    # st.write(accuracy)

    # 정확도를 계산하여 모델의 성능을 평가합니다.
    # accuracy = accuracy_score(y_test, test_pred_dt)


############ 그리드 ################
    # # 하이퍼파라미터 후보군 설정
    # param_grid = {
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.05, 0.1, 0.15]
    # }

    # # GridSearchCV를 이용한 하이퍼파라미터 튜닝
    # grid_search = GridSearchCV(
    #     estimator=lgb_model, 
    #     param_grid=param_grid,
    #     cv=5, # 교차 검증 횟수
    #     n_jobs=-1 # 모든 CPU 코어 사용
    # )

    # grid_search.fit(X_train, y_train)

    # # 최적의 하이퍼파라미터 조합
    # best_params = grid_search.best_params_
    # print(best_params)

    # # 최적의 모델
    # best_model = grid_search.best_estimator_    
    # accuracy = accuracy_score(y_test, best_model.predict(X_test))
    # st.write(accuracy)
############ 그리드 ################

    r1_col1, r1_col2, r1_col3, r1_col4  = st.columns(4)
    나이 = r1_col1.slider("나이",20,70,key="test21")
    일일급여 = r1_col2.slider("일일급여", 110, 1500,key="test22")
    회사와의거리 = r1_col3.slider("회사와의거리", 1, 30,key="test23")
    근무환경만족 = r1_col4.slider("근무환경만족", 1, 4,key="test24")

    # 두번째 행
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    시간당임금 = r2_col1.slider("시간당임금",30, 100,key="test25")
    직업만족도 = r2_col2.slider("직업만족도",1,4,key="test26")
    월수입 = r2_col3.slider('월수입',1000,20000,key="test27")
    이직회사수 = r2_col4.slider('이직회사수',0,9,key="test28")

    # 세번째 행
    r3_col1, r3_col2, r3_col3, r3_col4 = st.columns(4)
    급여인상비율 = r3_col1.slider("급여인상률",10,25,key="test29")
    동료관계만족도 = r3_col2.slider('동료관계만족도',1,4,key="test210")
    스톡옵션레벨 = r3_col3.slider('스톡옵션레벨',0,3,key="test211")
    워라벨 = r3_col4.slider('워라벨',1,4,key="test212")

    성별 = st.selectbox(
        '성별',
    ('남자', '여자'),key="test213")

    predict_button = st.button("퇴사유무 예측",key="test214")
    
    if predict_button:
            variable1 = np.array([나이, 일일급여, 회사와의거리, 근무환경만족, 성별=="남자", 시간당임금, 직업만족도, 월수입, 이직회사수, 급여인상비율, 동료관계만족도, 스톡옵션레벨, 워라벨])
            model1 = joblib.load('Data/pkl/LightGBM_model.pkl')
            pred1 = model1.predict([variable1])
            if pred1 == 1:
                st.write("퇴사")
            else:
                st.write("근속")

def xgBoost_model(df):
    with st.echo(code_location="below"):
        model_path = "Data/pkl/XGBoost.pkl"
        # model_path = "Data/pkl/GridedXGBoost.pkl" #그리드 pkl파일
        xgb_model = joblib.load(model_path)
        st.write("## XGBoost_model")
        st.write("**Gradient Boosting 알고리즘을 기반으로 하는 알고리즘**")
        st.write("이전 모델의 오차를 다음 모델이 보완하면서 학습을 진행합니다.")
        st.write("regularization 및 early stopping 기능을 제공하여 과적합을 방지할 수 있습니다.")
        

    # X,y = data_preprocessing(df)
    # # # 훈련 및 검증 데이터 분할
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # y_pred = xgb_model.predict(X_valid)
    # # # 정확도 계산

    # predict_button_dt3 = st.button('예측!')

    # if predict_button_dt3:        
    #     st.write(f'Train-set : {xgb_model.score(X_train, y_train)}')
    #     st.write(f'Test-set : {xgb_model.score(X_valid, y_valid)}')

    # # 정확도를 계산하여 모델의 성능을 평가합니다.
    # accuracy = accuracy_score(y_valid, y_pred)
    # st.write(accuracy)

############ 그리드 ################
#     # 훈련 및 검증 데이터 분할
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# # XGBClassifier 객체 생성
# xgb_model = XGBClassifier()

# # GridSearchCV를 위한 하이퍼 파라미터 설정
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.1, 0.01, 0.001],
#     'n_estimators': [100, 200, 300],
#     'min_child_weight': [1, 3, 5],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
# }

# # GridSearchCV 객체 생성
# grid = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=3)

# # GridSearchCV로 모델 훈련
# grid.fit(X_train, y_train)

# # 최적의 하이퍼 파라미터 출력
# print(f'Best parameter: {grid.best_params_}')

# # 최적의 모델로 예측 및 검증
# y_pred = grid.predict(X_valid)
# acc = accuracy_score(y_valid, y_pred)
# print(f'Accuracy: {acc}')
############ 그리드 ################   

    r1_col1, r1_col2, r1_col3, r1_col4  = st.columns(4)
    나이 = r1_col1.slider("나이",20,70,key="test31")
    일일급여 = r1_col2.slider("일일급여", 110, 1500,key="test32")
    회사와의거리 = r1_col3.slider("회사와의거리", 1, 30,key="test33")
    근무환경만족 = r1_col4.slider("근무환경만족", 1, 4,key="test34")

    # 두번째 행
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    시간당임금 = r2_col1.slider("시간당임금",30, 100,key="test35")
    직업만족도 = r2_col2.slider("직업만족도",1,4,key="test36")
    월수입 = r2_col3.slider('월수입',1000,20000,key="test37")
    이직회사수 = r2_col4.slider('이직회사수',0,9,key="test38")

    # 세번째 행
    r3_col1, r3_col2, r3_col3, r3_col4 = st.columns(4)
    급여인상비율 = r3_col1.slider("급여인상률",10,25,key="test39")
    동료관계만족도 = r3_col2.slider('동료관계만족도',1,4,key="test310")
    스톡옵션레벨 = r3_col3.slider('스톡옵션레벨',0,3,key="test311")
    워라벨 = r3_col4.slider('워라벨',1,4,key="test312")

    성별 = st.selectbox(
        '성별',
    ('남자', '여자'),key="test313")

    predict_button = st.button("퇴사유무 예측",key="test314")
    
    if predict_button:
            variable1 = np.array([나이, 일일급여, 회사와의거리, 근무환경만족, 성별=="남자", 시간당임금, 직업만족도, 월수입, 이직회사수, 급여인상비율, 동료관계만족도, 스톡옵션레벨, 워라벨])
            model1 = joblib.load('Data/pkl/XGBoost.pkl')
            pred1 = model1.predict([variable1])
            if pred1 == 1:
                st.write("퇴사")
            else:
                st.write("근속")
