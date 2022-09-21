import cv2
import numpy as np
import pandas as pd
import pytesseract

"""
    This File contains utility methods to support processes in the pipeline of Whiteboard Detection and text Extraction
"""


def get_track_bar_values():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src


def calc_biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder_points(my_points):
    my_points = my_points.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)

    myPointsNew[0] = my_points[np.argmin(add)]
    myPointsNew[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    myPointsNew[1] = my_points[np.argmin(diff)]
    myPointsNew[2] = my_points[np.argmax(diff)]

    return myPointsNew


def draw_rectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img


def morph_ops(img):
    """ Binarize image as a pre-requisite for applying pytesseract.
        To get binary image;
            1. Input Image is converted into greyscale,
            2. Dilation is applied to fill text,
            3. normalization is applied
            4. Adaptive Thresholding is applied
        """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=150)
    out_binary = cv2.adaptiveThreshold(out_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return out_binary


def morph_ops_with_warp(img):
    """ Binarize image as a pre-requisite for applying pytesseract.
    To get binary image;
        1. Input Image is converted into greyscale,
        2. Dilation is applied to fill text,
        3. normalization is applied
        4. Adaptive Thresholding is applied
        5. Applies Warping
    """
    widthImg = img.shape[1]
    heightImg = img.shape[0]

    # img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = get_track_bar_values()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    # FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = calc_biggest_contour(contours)  # FIND THE BIGGEST CONTOUR

    # If connected corners found -> capture image.
    if biggest.size != 0:
        biggest = reorder_points(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = draw_rectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg],
             [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg), cv2.INTER_AREA)

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 9, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image Array for Display
        img_all = ([img, imgGray, imgThreshold, imgContours],
                   [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

        return imgAdaptiveThre, img_all
    else:
        img_all = ([img, imgGray, imgThreshold, imgContours],
                   [imgBlank, imgBlank, imgBlank, imgBlank])
        return imgGray, img_all


def image_resize(img):
    """
    Resizes images, if app is being used on Mobile phones , as Images directly taken from camera are higher
    resolution and exhaust streamlit resources available on unpaid account
    Accepts image 'img' as numpy array
    """
    w = img.shape[0]
    h = img.shape[1]

    if w > 1500 or h > 1500:
        # As the model is crashing over big Images, resizing for memory optimization
        # Improve this by adding further refined aspect ratio resizing
        w_desired = 640
        h_desired = 640
        # w_factor = w / w_desired
        # h_factor = h / h_desired
        w_ratio = w_desired / w
        h_ratio = h_desired / h
        # img = cv2.resize(img, (round(w/w_factor), round(h / h_factor)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, None, fx=h_ratio, fy=w_ratio, interpolation=cv2.INTER_AREA)

    return img


def crop_image(img, annotations, margin=0):
    """
    Crops Image into sub images, based on whiteboard detection.
    Accepts original image and annotations predicted from Whiteboard Detectors
    """
    w = img.shape[1]
    h = img.shape[0]
    annotations = annotations[:, :4]
    annotations[:, 0] = annotations[:, 0] - margin
    annotations[:, 1] = annotations[:, 1] - margin
    annotations[:, 2] = annotations[:, 2] + margin
    annotations[:, 3] = annotations[:, 3] + margin

    annotations[np.where(annotations[:, 0] < 0)] = 0
    annotations[np.where(annotations[:, 1] < 0)] = 0
    annotations[np.where(annotations[:, 2] > w)] = w
    annotations[np.where(annotations[:, 3] > h)] = h

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
    relevant_classes = [16, 37, 101]
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
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    return image


def crop_txt(image, bbox, margin=0):
    w = image.shape[0]
    h = image.shape[1]
    imgs = []
    for bb in bbox:
        (tl, tr, br, bl) = bb
        x1 = np.max((int(tl[1]) - margin, 0))
        x2 = np.min((int(br[1]) + margin, w))
        y1 = np.max((int(tl[0]) - margin, 0))
        y2 = np.min((int(br[0]) + margin, h))

        imgs.append(image[x1:x2, y1:y2, :])
    return imgs


def extract_text_easyocr(img, reader):
    """
    Extracts text using easyocr.
    Limitation: Requires language type before extracting text. Fails if image is not aligned properly.
    Advantage: Encoder-Decoder+RCNN architecture takes care of processing the image before detection.
    Bounding Boxes provided for text detection  can be used with pytesseract, as an alternative to East Detector.
    """
    img = np.mean(img, axis=2).astype('uint8')
    annot_txt = reader.readtext(img, paragraph="False")
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
    binary_img, _ = morph_ops_with_warp(img)
    txt = pytesseract.image_to_string(binary_img, config=' --oem 1 ')
    # txt = pytesseract.image_to_string(binary_img, config='--psm 12 --oem 1 ')

    return binary_img, txt


def easy_pytesseract_ocr(img, reader, preprocessing_method=0):
    """Uses Pytesseract to detect text on Text Bounding boxes detected by easy ocr."""
    img_easy = np.mean(img, axis=2).astype('uint8')
    annot_txt = reader.readtext(img_easy, paragraph="False")
    annot = [annot for (annot, text) in annot_txt]

    # extract sub images against all the bounding boxes.
    bbx_imgs = crop_txt(img, annot, margin=0)
    txts = []
    binary_imgs = []
    for bbx_img in bbx_imgs:
        if preprocessing_method == 0:
            binary_img = morph_ops(bbx_img)
        else:
            binary_img, _ = morph_ops_with_warp(bbx_img)
        binary_imgs.append(binary_img)
        txt = pytesseract.image_to_string(binary_img, config='--psm 12 --oem 1 ')
        txts.append(txt)

    return binary_imgs, txts
