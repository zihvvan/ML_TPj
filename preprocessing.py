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

def pre_processing(df):
    # 필요없는 Columns 드랍
    df1 = df.drop(['school','classroom','student_id'], axis=1)
    # 원 핫 인코딩으로 분류형 데이터 처리
    df1 = pd.get_dummies(df1, columns=['school_setting','school_type','teaching_method','gender','lunch'], drop_first=True)
    df2 = df1.rename(columns={'n_student' : '학생수', 'pretest' : '사전점수', 'posttest': '시험점수', 'school_setting_Suburban':'Suburban', 'school_setting_Urban':'Urban', 'school_type_Public':'Public', 'teaching_method_Standard':'Standard', 'gender_Male':'Male','lunch_Qualifies for reduced/free lunch':'free lunch'})
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