import streamlit as st
from PIL import Image
import view
from datetime import datetime, timedelta
def cal_time():
    today = datetime.today()

    after_100_days = today + timedelta(days=100)

    countdown = after_100_days - today

    return countdown.days

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
        st.markdown("![Git](https://img.shields.io/badge/git-444444?style=for-the-badge&logo=git)[Git](https://github.com/mastgm0817/ML_TPrj)")
        st.markdown("![Notion](https://img.shields.io/badge/Notion-444444?style=for-the-badge&logo=Notion)[Notion](http://bit.ly/3lSMdPR)")

    if add_selectbox == "성적 예측 모델":
        view.view_model1()
    else:
        view.view_model2()


main()