import streamlit as st
import pandas as pd

def s_pre_processing(df):
    # 필요없는 Columns 드랍
    df1 = df.drop(['school','classroom','student_id'], axis=1)
    # 원 핫 인코딩으로 분류형 데이터 처리
    df1 = pd.get_dummies(df1, columns=['school_setting','school_type','teaching_method','gender','lunch'], drop_first=True)
    df2 = df1.rename(columns={'n_student' : '학생수', 'pretest' : '사전점수', 'posttest': '시험점수', 'school_setting_Suburban':'Suburban', 'school_setting_Urban':'Urban', 'school_type_Public':'Public', 'teaching_method_Standard':'Standard', 'gender_Male':'Male','lunch_Qualifies for reduced/free lunch':'free lunch'})
    return df2

def a_pre_processing(df):
    df1 = df.drop(['BusinessTravel','Department','Education','EducationField','EmployeeCount','EmployeeNumber','JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus',  'MonthlyRate', 'Over18', 'OverTime', 'PerformanceRating','StandardHours', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'],axis=1) # 필요없는 feature 삭제
    df2 = df1.loc[:,['Age','DailyRate','DistanceFromHome','EnvironmentSatisfaction','Gender','HourlyRate', 'JobSatisfaction','MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel','WorkLifeBalance','Attrition']] # 종속변수값 재배치
    return df2

def load_data(choose):
    # csv데이터 불러오기
    if choose == 1:
        data_url = "Data/test_scores.csv"
        df = pd.read_csv(data_url) # URL로 CSV 불러오기
    else:
        data_url = "Data/HR_Analytics.csv"
        df = pd.read_csv(data_url) # URL로 CSV 불러오기
    return df