import cv2
import numpy as np
import json
import functools
from pyzbar import pyzbar
import itertools

from .utils import listit


BORDER_LEFT, BORDER_RIGHT = (
    np.array([1001.05364189, 1140.30542526, 1279.14998283, 1419.51725082,
              1559.92500267, 1703.37992821, 1962.73085518, 2108.28859329,
              2253.78823833, 2398.27252483, 2543.75992908, 2687.42875385]),
    np.array([1119.95232735, 1258.48413773, 1398.24998055, 1538.33389664,
              1679.02499733, 1822.78003273, 2081.88634453, 2227.74712448,
              2372.72135639, 2517.4570406,  2663.21084728, 2806.73113384])
)
BORDER_TOP, BORDER_BOTTOM = (
    np.array([117.07986768,  262.0833257,   406.66667048,  551.08331553,
              696.61630789,  841.62500191,  988.26667436,  1134.61490504,
              1278.74997139, 1424.70830472, 1570.14406967, 1715.04166921,
              1860.08333524, 2003.87500127, 2147.95829391, 2294.20823479,
              2440.94248358, 2589.62494087, 2736.86812528, 2881.99990145]),
    np.array([232.93112373,  377.99998728,  522.49999619,  667.16664378,
              813.08647124,  958.37499809,  1105.2069788,  1250.93129857,
              1395.41663424, 1541.20830091, 1686.43419774, 1830.95833079,
              1975.91666476, 2120.0416654,  2263.37495804, 2409.6248951,
              2557.37034194, 2705.20827039, 2852.98624484, 2998.16656176])
)
LABELS = ('A', 'B', 'C', 'D', 'E', 'F')
QUESTIONS = tuple(range(1, 41))
WIDTH = 3000
HEIGHT = int(WIDTH * 578 / 403)


@functools.lru_cache(1)
def _get_small_rectangles_positions_middle():
    xl, yt = np.meshgrid(BORDER_LEFT, BORDER_TOP)
    xr, yb = np.meshgrid(BORDER_RIGHT, BORDER_BOTTOM)
    xc, yc = (xl+xr)/2, (yt+yb)/2
    dx, dy = (xl-xr), (yb-yt)

    labels = LABELS
    result = []
    for i, j in itertools.product(range(xc.shape[0]), range(xc.shape[1])):
        question = QUESTIONS[i + (len(QUESTIONS)//2) * (j // len(LABELS))]
        label = labels[j % len(LABELS)]
        box = (question, label), ((yc[i, j], xc[i, j]), (dy[i, j], dx[i, j]), 0.)
        result.append(box)
    return tuple(result)


RECTANGLES_POSITIONS_MIDDLE = dict(_get_small_rectangles_positions_middle())


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
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    allowed_boxes = []
    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # no noise condition
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

    result = cv2.resize(result, (WIDTH, HEIGHT))
    return result


def get_small_rectangles(image, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edged = cv2.Canny(gray, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    allowed_boxes = []
    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # no small noise condition
        if (rect[1][0] > 50) and (rect[1][1] > 50):
            # get rid off "empty squares"
            statistics = get_statistics(image, box)
            if (np.mean(statistics) < 200) and (
                (rect[0][1] > ylim[0]) and (rect[0][1] < ylim[1])
                and
                (rect[0][1] > xlim[0]) and (rect[0][1] < xlim[1])
            ):
                rect = listit(rect)
                rect[1][0] += 15
                rect[1][1] += 15
                allowed_boxes.append(tuple(rect))

    return allowed_boxes


def validate_qr_code(image):
    barcodes = pyzbar.decode(image)
    return json.loads(barcodes[0].data.decode("utf-8"))


class CroppedAnswers(object):
    def __init__(self, cropped, validate_qr=False):
        if validate_qr:
            validate_qr_code(cropped)
        self.cropped = cropped

    @classmethod
    def from_raw(cls, image, **kwargs):
        return cls(crop_image(image), **kwargs)

    @classmethod
    def from_file(cls, path, cropped=True, **kwargs):
        if cropped:
            return cls(cv2.imread(path), **kwargs)
        else:
            return cls.from_raw(cv2.imread(path), **kwargs)

    def __array__(self, dtype=None):
        if dtype:
            return np.cast(self.cropped, out=self.cropped, dtype=dtype)
        else:
            return self.cropped

    def get_small_rectangles_positions_bottom(self):
        """Experimental"""
        return get_small_rectangles(
            self.cropped,
            ylim=(3000, np.inf)
        )

    def recognized_rectangles_image(self):
        image = self.cropped.copy()
        for _, box_spec in _get_small_rectangles_positions_middle():
            box = cv2.boxPoints(box_spec)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 0, 255), 4)
        return image
