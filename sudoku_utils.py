import cv2
import numpy as np
from tensorflow.python.keras.models import load_model


# READ THE MODEL WEIGHTS
def initialize_prediction_model():
    model = load_model('Resources/myModel.h5')
    return model


# 1 - Preprocessing Image
def pre_process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return img_threshold


# 3 - Reorder points for Warp Perspective
def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new


# 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# 4 - TO SPLIT THE IMAGE INTO 81 DIFFERENT IMAGES
def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# 4 - GET PREDICTIONS ON ALL IMAGES
def get_prediction(boxes, model):
    result = []
    for image in boxes:
        # PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        # GET PREDICTION
        predictions = model.predict(img)
        class_index = model.predict_classes(img)
        probability_value = np.amax(predictions)
        # SAVE TO RESULT
        if probability_value > 0.8:
            result.append(class_index[0])
        else:
            result.append(0)
    return result


# 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def display_numbers(img, numbers, color=(0, 255, 0)):
    sec_w = int(img.shape[1] / 9)
    sec_h = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (x * sec_w + int(sec_w / 2) - 10, int((y + 0.8) * sec_h)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


# 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def draw_grid(img):
    sec_w = int(img.shape[1] / 9)
    sec_h = int(img.shape[0] / 9)
    for i in range(0, 9):
        pt1 = (0, sec_h * i)
        pt2 = (img.shape[1], sec_h * i)
        pt3 = (sec_w * i, 0)
        pt4 = (sec_w * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img


# 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stack_images(img_array, scale):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        hor_con = np.concatenate(img_array)
        ver = hor
    return ver
