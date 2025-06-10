import cv2
import numpy as np
import os
import suduko_solver
from sudoku_utils import pre_process, biggest_contour, reorder, get_prediction, split_boxes, display_numbers, \
    draw_grid, stack_images, initialize_prediction_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_path = "Resources/asd.jpeg"
img_height = 450
img_width = 450

# 1. PREPARE THE IMAGE
img = cv2.imread(image_path)
img = cv2.resize(img, (img_width, img_height))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
blank_img = np.zeros((img_height, img_width, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
threshold_img = pre_process(img)

# 2. FIND ALL CONTOURS
contours_img = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
big_contour_img = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)  # DRAW ALL DETECTED CONTOURS

# 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
biggest, max_area = biggest_contour(contours)  # FIND THE BIGGEST CONTOUR
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(big_contour_img, biggest, -1, (0, 0, 255), 25)  # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    warp_colored_img = cv2.warpPerspective(img, matrix, (img_width, img_height))
    detected_digits_img = blank_img.copy()
    warp_colored_img = cv2.cvtColor(warp_colored_img, cv2.COLOR_BGR2GRAY)

    # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    solved_digits_img = blank_img.copy()
    boxes = split_boxes(warp_colored_img)
    print(len(boxes))
    # cv2.imshow("Sample",boxes[65])
    model = initialize_prediction_model()
    numbers = get_prediction(boxes, model)
    print(numbers)
    detected_digits_img = display_numbers(detected_digits_img, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    # 5. FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers, 9)
    print(board)
    try:
        suduko_solver.solve(board)
    except:
        pass

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    solved_digits_img = display_numbers(solved_digits_img, solvedNumbers)

    # # 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts1 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    img_inv_warp_colored = cv2.warpPerspective(solved_digits_img, matrix, (img_width, img_height))
    inv_perspective = cv2.addWeighted(img_inv_warp_colored, 1, img, 0.5, 1)
    detected_digits_img = draw_grid(detected_digits_img)
    solved_digits_img = draw_grid(solved_digits_img)

    imageArray = ([img, threshold_img, contours_img, big_contour_img],
                  [detected_digits_img, solved_digits_img, inv_perspective, blank_img])
    stackedImage = stack_images(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv2.waitKey(0)
