import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

st.title('Whiteboard Detection and Text Identification')

st.header('Upload Image from drive for whiteboard detection')

im = 'https://ultralytics.com/images/zidane.jpg'
@st.cache
def load_model():
    _model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return _model


model = load_model()
uploaded_img = st.file_uploader('Upload Image')
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    fig = plt.figure()
    plt.imshow(img)
    st.pyplot(fig)

    # run the uploaded image through model to see how it performs on custom pretrained weights
    results = model(img)

    st.write(results)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    # st.write(results.pandas().xyxy[0])
    st.pyplot(fig)
