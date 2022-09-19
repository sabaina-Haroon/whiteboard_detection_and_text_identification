import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import easyocr
from util import *
from PIL import Image
import matplotlib.cm as cm
st.title('Whiteboard Detection and Text Identification')

def plot_imgs(images, is_binary=False):

    fig = plt.figure(figsize=(20, 12))
    v = 330
    for i in (range(len(images))):
        v = v + 1
        ax1 = fig.add_subplot(v)
        if is_binary:
            ax1.imshow(images[i], cmap=cm.gray)
        else:
            ax1.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax1.axis('off')
    return fig



@st.cache
def load_model():
    _model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m_Objects365.pt')
    return _model


@st.cache
def load_ocrmodel():
    _reader = easyocr.Reader(['en'], model_storage_directory='./ocr_models/', detector='DB', recognizer='Transformer')
    return _reader


model = load_model()
reader_easyocr = load_ocrmodel()
st.header('Upload Image for whiteboard detection')
uploaded_img = st.file_uploader('Upload Image')

if uploaded_img is not None:
    st.header('Whiteboard Detection')
    # img = Image.open(uploaded_img)
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # img = cv2.imread(uploaded_img)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('Uploaded Image', fontsize=30)
    img_yolo = img.copy()
    # run the uploaded image through model to see how it performs on custom pretrained weights
    results = model(img_yolo)
    ax[1].imshow(cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB))
    ax[1].axis('off')
    ax[1].set_title('Detection Results', fontsize=30)
    st.pyplot(fig)
    st.write(results)

    st.subheader('Region of Interest')
    pred_annot = filter_annotations(results.pred)
    # clipped_images = crop_image(np.array(img), pred_annot)
    clipped_images = crop_image(img, pred_annot)

    st.pyplot(plot_imgs(clipped_images))

    tab1, tab2, tab3 = st.tabs(["Pytesseract", "EasyOCR", "EasyOCR + Pytesseract"] )

    with tab1:
        pytesseract_output = [extract_text_tesseract(x) for x in clipped_images]
        binary_imgs = [pytesseract_output[i][0] for i in range(len(pytesseract_output))]
        st.markdown('#### Processed Images as Input to pytesseract')
        st.pyplot(plot_imgs(binary_imgs, is_binary=True))
        st.markdown('#### Text Extraction')
        txt_pytesseract = [pytesseract_output[i][1] for i in range(len(pytesseract_output))]
        c= [st.write(i) for i in txt_pytesseract]

    with tab2:
        easyocr_output = [extract_text_easyocr(x, reader_easyocr) for x in clipped_images]
        bb_imgs = [easyocr_output[i][0] for i in range(len(easyocr_output))]
        st.markdown('#### Text Detection on Input to Easyocr')
        st.pyplot(plot_imgs(bb_imgs, is_binary=True))
        st.markdown('#### Text Extraction')
        txt_easyocr = [easyocr_output[i][1] for i in range(len(easyocr_output))]
        c = [st.write(i) for i in txt_easyocr]



