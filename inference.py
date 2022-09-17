import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
import cv2
import easyocr
import matplotlib
import numpy as np
# matplotlib.use('tkagg')

st.title('Whiteboard Detection and Text Identification')

st.header('Upload Image from drive for whiteboard detection')

im = 'https://ultralytics.com/images/zidane.jpg'

@st.cache
def load_model():
    _model = torch.hub.load('ultralytics/yolov5','custom' , path='yolov5m_Objects365.pt')
    return _model

@st.cache
def load_ocrmodel():
    _reader = easyocr.Reader(['en'], model_storage_directory='./ocr_models/', detector='DB', recognizer='Transformer')
    return _reader

model = load_model()
reader = load_ocrmodel()


uploaded_img = st.file_uploader('Upload Image')
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    w = img.size[0]
    h = img.size[1]
    # As the model is crashing over big Images from mobile devices, resizing for memory optimization
    while w > 640 and h > 640:
        # To be done , Improve this by adding aspect ratio resizing
        img = cv2.resize(np.array(img), (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
        w = img.shape[0]
        h = img.shape[1]

    fig = plt.figure()
    plt.imshow(img)
    st.pyplot(fig)

    # run the uploaded image through model to see how it performs on custom pretrained weights
    results = model(img)
    results1 = model(img) # just checking deployment issue

    st.write(results)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    # st.write(results.pandas().xyxy[0])
    st.pyplot(fig)

    result = reader.readtext(np.array(img),detail=0, paragraph="False")
    st.header('Text Detected')
    st.write(result)
