import cv2
import numpy as np


def get_statistics(image, approx):
    approx = approx.tolist()

    approx = sorted(approx, key=lambda x: x[0])
    xmin = approx[0][0]
    xmax = approx[-1][0]

    approx = sorted(approx, key=lambda x: x[1])
    ymin = approx[0][1]
    ymax = approx[-1][1]
    black_square = image[ymin:ymax, ::][:, xmin:xmax]

    lower = np.array([230, 230, 230])
    upper = np.array([256, 256, 256])
    shape_mask = cv2.inRange(black_square, lower, upper)

    return shape_mask


def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edged = cv2.Canny(gray, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("closed.jpg", closed)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    allowed_boxes = []
    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Условие на отсев шума
        if (rect[1][0] > 40) and (rect[1][1] > 40):
            statistics = get_statistics(image, box)
            if np.mean(statistics) < 100:
                allowed_boxes.append(rect)

    nec_boxes = [min(allowed_boxes, key=lambda x: x[0][0] + x[0][1]),
                 max(allowed_boxes, key=lambda x: x[0][0] + x[0][1]),
                 min(allowed_boxes, key=lambda x: x[0][0] - x[0][1]),
                 max(allowed_boxes, key=lambda x: x[0][0] - x[0][1])]

    large = sorted(image.shape)[-1]
    small = sorted(image.shape)[-2]
    pts1 = np.float32([i[0] for i in nec_boxes])
    pts2 = np.float32([[0, 0], [large, small], [0, small], [large, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(image, M, (large, small))

    width = 3000
    result = cv2.resize(result, (width, int(width * 578 / 403)))

    return result
