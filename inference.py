import easyocr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import streamlit as st
import torch

from util import *

st.set_page_config(
    page_title='Whiteboard Detection and Text Identification'
)
st.title("Whiteboard Detection and Text Identification")


def plot_imgs(images, is_binary=False):
    fig_ = plt.figure(figsize=(20, 10))
    v = 330
    for i in (range(len(images))):
        v = v + 1
        if v % 10 == 0:
            v = v + 1
        ax1 = fig_.add_subplot(v)
        if is_binary:
            ax1.imshow(images[i], cmap=cm.gray)
        else:
            ax1.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax1.axis('off')
    return fig_


@st.cache
def load_model(model_type='Pretrained'):
    if model_type == 'Pretrained':
        _model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m_Objects365.pt')
    # elif model_type == 'Transfer Learning':
    #     _model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    return _model


@st.cache
def load_ocrmodel():
    _reader = easyocr.Reader(['en'], model_storage_directory='./ocr_models/', detector='DB', recognizer='Transformer')
    return _reader


@st.cache(ttl=600, max_entries=5)
def process_tesseract(clipped_images_):
    pytesseract_output = [extract_text_tesseract(x) for x in clipped_images_]
    binary_imgs_ = [pytesseract_output[i][0] for i in range(len(pytesseract_output))]
    txt_pytesseract_ = [pytesseract_output[i][1] for i in range(len(pytesseract_output))]
    return binary_imgs_, txt_pytesseract_


@st.cache(ttl=600, max_entries=5)
def process_easyocr(clipped_images_):
    easyocr_output = [extract_text_easyocr(x, reader_easyocr) for x in clipped_images_]
    bb_imgs_ = [easyocr_output[i][0] for i in range(len(easyocr_output))]
    txt_easyocr_ = [easyocr_output[i][1] for i in range(len(easyocr_output))]
    return bb_imgs_, txt_easyocr_


@st.cache(ttl=600, max_entries=5)
def process_easytesseract(clipped_images_, preprocessing=0):
    binary_imgs_all = []
    txts_all = []
    for clipped_image in clipped_images_:
        binary_imgs_, txt_ = easy_pytesseract_ocr(clipped_image, reader_easyocr, preprocessing)
        if len(binary_imgs_) > 0:  # Handling cases where Easyocr does not detect any bounding box
            binary_imgs_all.append(binary_imgs_)
            txts_all.append(txt_)
    return binary_imgs_all, txts_all


@st.cache(ttl=600, max_entries=5)
def load_image(uploaded_img_):
    file_bytes = np.asarray(bytearray(uploaded_img_.read()), dtype=np.uint8)
    img_ = cv2.imdecode(file_bytes, 1)
    return img_


# model_type = st.selectbox('Select the model', ['Pretrained', 'Transfer Learning'])

model = load_model()
reader_easyocr = load_ocrmodel()
# st.header('Upload Image for whiteboard detection')
st.markdown('## <b><u><font color="Green">Upload Image for whiteboard detection </font></b></u>',
            unsafe_allow_html=True)
uploaded_img = st.file_uploader('Upload Image')
if uploaded_img is not None:
    st.markdown('## <b><u><font color="Green">Whiteboard Detection </font></b></u>', unsafe_allow_html=True)
    img = load_image(uploaded_img)
    img = image_resize(img)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('Uploaded Image', fontsize=8)
    img_yolo = img.copy()
    # run the uploaded image through model to see how it performs on custom pretrained weights
    results = model(img_yolo)
    ax[1].imshow(cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB))
    ax[1].axis('off')
    ax[1].set_title('Detection Results', fontsize=8)
    st.pyplot(fig)

    pred_annot = filter_annotations(results.pred)

    input_imgs = []
    if len(pred_annot) > 0:
        st.markdown('### <u> Region of Interest </u>', unsafe_allow_html=True)
        # clipped_images = crop_image(np.array(img), pred_annot)
        clipped_images = crop_image(img, pred_annot)
        st.pyplot(plot_imgs(clipped_images))
        input_imgs = clipped_images
    else:
        st.markdown("### <center> <font color='brown'>If you have uploaded a whiteboard image and Detector has not "
                    "detected it, It might be because the image is relatively zoomed-in. Therefore, In this case, "
                    "we will pass the entire image to the OCR algorithm.</font></center>", unsafe_allow_html=True)
        input_imgs = [img]
    st.markdown('## <b><u><font color="Green">Text Identification</font></b></u>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Pytesseract", "EasyOCR", "EasyOCR + Pytesseract"])
    with tab1:
        st.markdown('###### <font color="Red">Extracts text using pytesseract</font>', unsafe_allow_html=True)
        st.markdown('<b><font color="Red">Limitations:</b> </font>It requires the image to be processed and binarized '
                    'before text identification. Whereas Binarization step is specific to the environment in '
                    'which the image is captured and difficult to generalize.',
                    unsafe_allow_html=True)
        st.markdown('<b><font color="Red">Advantage:</b></font> Light-weight compared to easyocr and supports '
                    'multiple languages at once.', unsafe_allow_html=True)
        binary_imgs, txt_pytesseract = process_tesseract(input_imgs)
        st.markdown('#### <u>Processed Images as Input to pytesseract</u>', unsafe_allow_html=True)
        st.pyplot(plot_imgs(binary_imgs, is_binary=True))
        st.markdown('#### <u>Text Extraction</u>', unsafe_allow_html=True)
        c = [st.write(i) for i in txt_pytesseract]
    with tab2:
        st.markdown('###### <font color="Red">Extracts text using Easyocr.</font>', unsafe_allow_html=True)
        st.markdown('<b><font color="Red">Limitations:</b> </font>It requires language type before extracting text. '
                    'It supports around 80 languages, whereas models for most languages are computationally expensive '
                    'to maintain in deployment.',
                    unsafe_allow_html=True)
        st.markdown('<b><font color="Red">Advantage:</b></font> Encoder-Decoder+RCNN architecture takes care of '
                    'processing the image before detection. Bounding Boxes provided for text detection  can be used '
                    'with pytesseract, as an alternative to East Detector.', unsafe_allow_html=True)
        bb_imgs, txt_easyocr = process_easyocr(input_imgs)
        st.markdown('#### <u>Text Detection on Input to Easyocr</u>', unsafe_allow_html=True)
        st.pyplot(plot_imgs(bb_imgs, is_binary=True))
        st.markdown('#### <u>Text Extraction</u>', unsafe_allow_html=True)
        c = [st.write(i) for i in txt_easyocr]
    with tab3:
        st.markdown('<font color="Red">This method uses text detection capability from EasyOCR. Based on the bounding '
                    'boxes marked by EasyOCR, we individually apply pytesseract to subimages. .</font>',
                    unsafe_allow_html=True)
        method1, method2 = st.tabs(['Preprocessing Method 1', 'Preprocessing Method 2'])
        with method1:
            binary_imgs, txt = process_easytesseract(input_imgs, preprocessing=0)
            st.markdown('#### <u>Clipped Images from easyocr as Input to pytesseract</u>', unsafe_allow_html=True)
            for binary_img in binary_imgs:
                st.pyplot(plot_imgs(binary_img, is_binary=True))
            st.markdown('#### <u>Text Extraction</u>', unsafe_allow_html=True)
            st.write(txt)
        with method2:
            binary_imgs, txt = process_easytesseract(input_imgs, preprocessing=1)
            st.markdown('#### <u>Clipped Images from easyocr as Input to pytesseract</u>', unsafe_allow_html=True)
            for binary_img in binary_imgs:
                st.pyplot(plot_imgs(binary_img, is_binary=True))
            st.markdown('#### <u>Text Extraction</u>', unsafe_allow_html=True)
            st.write(txt)
