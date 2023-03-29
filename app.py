import streamlit as st
from PIL import Image
import view
from datetime import date
def cal_time():
    today = date.today()
    future_date = date(2023, 8, 24)
    delta = future_date - today

    return delta.days

def main():
    image1 = Image.open('image/main.webp')
    st.image(image1, width=600)
    
    with st.sidebar:
        add_selectbox = st.sidebar.selectbox(
            "모델을 선택하세요. ",
            ("성적 예측 모델", "회사퇴사 예측 모델")
        )
        days = cal_time()
        st.markdown(f"<div style='font-weight:bold; font-size:40px; text-align:center'>D - {days}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight:bold; font-size:40px; text-align:center'>Quit? or Not!</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"<div align=center><h1>📚 Reference </h1></div>", unsafe_allow_html=True)
        st.markdown(":point_down:")
        st.markdown("[![Git](https://img.shields.io/badge/git-444444?style=for-the-badge&logo=git)](https://github.com/mastgm0817/ML_TPrj)")
        st.markdown("[![Notion](https://img.shields.io/badge/Notion-444444?style=for-the-badge&logo=Notion)](https://bit.ly/3lSMdPR)")
    if add_selectbox == "성적 예측 모델":
        view.view_model1()
    else:
        view.view_model2()


main()