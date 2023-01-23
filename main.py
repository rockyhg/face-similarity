import streamlit as st
from PIL import Image

st.title('顔 類似度アプリ')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
