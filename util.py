import cv2
import numpy as np
import pandas as pd
import pytesseract

"""
    This File contains utility methods to support processes in the pipeline of Whiteboard Detection and text Extraction
"""


def morphological_ops(image):
    """ Binarize image as a pre-requisite for applying pytesseract.
    To get binary image;
        1. Input Image is converted into greyscale,
        2. Dilation is applied to fill text,
        3. normalization is applied
        4. Adaptive Thresholding is applied
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=150)
    out_binary = cv2.adaptiveThreshold(out_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return out_binary
    # plt.imshow(out_binary, cmap=cm.gray)
    # txt = pytesseract.image_to_string(out_binary, config='--oem 1 ')
    # print(txt)


def image_resize(img):
    """
    Resizes images, if app is being used on Mobile phones , as Images directly taken from camera are higher
    resolution and exhaust streamlit resources available on unpaid account
    Accepts image 'img' as numpy array
    """
    w = img.size[0]
    h = img.size[1]
    # As the model is crashing over big Images from mobile devices, resizing for memory optimization
    while w > 640 and h > 640:
        # To be done , Improve this by adding aspect ratio resizing
        # img = cv2.resize(np.array(img), (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        w = img.shape[0]
        h = img.shape[1]
    return img


def crop_image(img, annotations):
    """
    Crops Image into sub images, based on whiteboard detection.
    Accepts original image and annotations predicted from Whiteboard Detectors
    """
    annotations = annotations[:, :4]
    annotations = annotations.astype('int')
    print(annotations)
    clipped_images = [img[x[1]:x[3], x[0]:x[2]] for x in annotations]
    return clipped_images


def filter_annotations(predictions):
    """
    Filters annotations for only the relevant classes.
    Yolov5 detection model, detects multiple objects in an image based on the total number of classes available.
    Yolov5 trained on Object365 data is used as pretrained model for detection.
    """
    #   16: Picture/Frame, 37: Monitor / TV,   101: Blackboard/Whiteboard,   218: Projector
    relevant_classes = [16, 37, 101, 218]
    annotations = [x.cpu().numpy() for x in predictions]  # Currently FE app designed for single image
    annotations = np.concatenate(annotations, axis=0)
    annotations = pd.DataFrame(annotations)
    # Yolov5 returns predictions in the format of x1,y1,x2,y2,confidence,class_label.
    annotations = annotations[annotations[5].isin(relevant_classes)].reset_index(drop=True)
    annotations = annotations.values
    return annotations


def draw_bbx(image, bbox):
    for bb in bbox:
        (tl, tr, br, bl) = bb
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    return image


def extract_text_easyocr(img, reader):
    """
    Extracts text using easyocr.
    Limitation: Requires language type before extracting text. Fails if image is not aligned properly.
    Advantage: Encoder-Decoder+RCNN architecture takes care of processing the image before detection.
    Bounding Boxes provided for text detection  can be used with pytesseract, as an alternative to East Detector.
    """
    img = np.mean(img, axis=2).astype('uint8')
    annot_txt = reader.readtext(img,  paragraph="False")
    annot = [annot for (annot, text) in annot_txt]
    image_bb = draw_bbx(img, annot)
    txt = [text for (annot, text) in annot_txt]
    return image_bb, txt


def extract_text_tesseract(img):
    """
    Extracts text using pytesseract.
    Limitation: It requires image to be processed and binarized before text identification.
                Binarization step is specific to the environment in which image is captured.
    Advantage: Light-weight compared to easyocr.
    """
    binary_img = morphological_ops(img)
    txt = pytesseract.image_to_string(binary_img, config=' --oem 1 ')
    # txt = pytesseract.image_to_string(binary_img, config='--psm 12 --oem 1 ')

    return binary_img, txt
