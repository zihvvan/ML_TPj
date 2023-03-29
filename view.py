import streamlit as st
import pandas as pd
import data_preprocess_score, data_preprocess_attrition, visualization_process_score, visualization_process_attrition, models
import pre_process
import decision_tree

def view_model1():
    st.title("다중선형회귀 vs 다항선형회귀(Lasso)")
    tab1, tab2, tab3 = st.tabs(["성적 예측","데이터셋 지표분석","시각화"])
    df = pre_process.load_data(1)
    with tab1:
            st.header("성적 예측 모델")
            models.linear_model()
    with tab2:
            st.header("원본 데이터")
            st.write(df)
            st.write("---")
            st.header("데이터셋 통계자료 ")
            st.write("데이터셋 전체에 대한 통계 정보를 확인합니다.")
            processed_df = pre_process.pre_processing(df)
            visualization_process.describe_linear_model(df)
            st.write("---")
            st.header("데이터셋 Drop & One-Hot Enconding")
            st.write("불필요한 열 제거, 결측지 처리, 범주형 변수 처리등의 이유로 데이터 전처리 과정에서")
            st.write("분석의 효율성을 높이기 위해 필요한 과정입니다.")
            st.write(processed_df)
            st.write("---")
            st.header("Min-Max Scaling")
            st.write("데이터 정규화, 이상치 처리, 머신 러닝 모델 성능 향상을 위한 데이터 전처리 기법")
            data_frame2 = data_preprocess.linear_process(df)
            s_df, comparison, data_frame1 = data_preprocess.poly_model(df)
            data_preprocess.draw_table(data_frame1, data_frame2 )
    with tab3:
            st.header("시각화")
            visualization_process.visualization(df, s_df, comparison)



def view_model2():
    st.title("회사퇴사 예측 모델")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["RandomForest", "XGBoost", 'LightGBM','데이터셋', '결정트리 시각화'])
    df = pre_process.load_data(2)
    with tab1:
            # RandomForest Model
            models.random_forest_model(df)
    with tab2:
            # XGBoost Model
            models.xgBoost_model(df)
    with tab3:
            # LightGBM Model
            models.lightGBM_model(df)
    with tab4:
            st.header("원본 데이터")
            st.write(df)
            st.header("데이터셋 통계자료 ")
            visualization_process_attrition.describe_attrition_model(df)
            df1 = pre_process.a_pre_processing(df)
            st.header("데이터셋 Drop & One-Hot Enconding")
            X,y = data_preprocess_attrition.make_dummies(df1)
            st.write(X)
            st.header("모델별 지표 분석")
            data_preprocess_attrition.create_table()
    with tab5:
            df = pre_process.load_data(2)
            decision_tree.decision_tree_preprocessing(df)

            