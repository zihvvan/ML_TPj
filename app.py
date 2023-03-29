import streamlit as st
from PIL import Image
import view

def main():
    image1 = Image.open('image/m_img.png')
    st.image(image1, width=600)

    add_selectbox = st.sidebar.selectbox(
        "모델을 선택하요. ",
        ("성적 예측 모델", "회사퇴사 예측 모델")
    )

    if add_selectbox == "성적 예측 모델":
        view.view_model1()
    else:
        view.view_model2()


main()