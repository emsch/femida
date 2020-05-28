import cv2
import numpy as np
import json
import functools
import collections
import itertools
import os

from .utils import listit


if "FEMIDA_OCR_BORDER_LEFT" not in os.environ:
    BORDER_LEFT = np.array(
        [
            198,
            332,
            461,
            595,
            727,
            857,
            989,
            1117,
            1252,
            1384,
            1516,
            1651,
            1783,
            1914,
            2047,
            2176,
            2309,
            2442,
            2573,
            2706,
        ]
    )
else:
    BORDER_LEFT = np.array(
        list(map(int, os.environ["FEMIDA_OCR_BORDER_LEFT"].split(",")))
    )

if "FEMIDA_OCR_UPDATES_BORDER_LEFT" not in os.environ:
    BORDER_UPDATES_LEFT = np.array(
        [
            485,
            593,
            715,
            838,
            960,
            1080,
            1200,
            1560,
            1663,
            1783,
            1906,
            2030,
            2145,
            2270,
        ]
    )
else:
    BORDER_UPDATES_LEFT = np.array(
        list(map(int, os.environ["FEMIDA_OCR_UPDATES_BORDER_LEFT"].split(",")))
    )

if "FEMIDA_OCR_VARIANT_BORDER_LEFT" not in os.environ:
    BORDER_VARIANT_LEFT = np.array([450, 565, 687, 810, 937])
else:
    BORDER_VARIANT_LEFT = np.array(
        list(map(int, os.environ["FEMIDA_OCR_VARIANT_BORDER_LEFT"].split(",")))
    )

MARGIN_HORIZONTAL = int(os.environ.get("FEMIDA_OCR_MARGIN_HORIZONTAL", 84))

BORDER_RIGHT = BORDER_LEFT + MARGIN_HORIZONTAL
BORDER_UPDATES_RIGHT = BORDER_UPDATES_LEFT + MARGIN_HORIZONTAL
BORDER_VARIANT_RIGHT = BORDER_VARIANT_LEFT + MARGIN_HORIZONTAL

if "FEMIDA_OCR_BORDER_TOP" not in os.environ:
    BORDER_TOP = np.array([1322, 1450, 1580, 1715, 1843, 2087, 2217, 2349, 2479, 2610])
else:
    BORDER_TOP = np.array(
        list(map(int, os.environ["FEMIDA_OCR_BORDER_TOP"].split(",")))
    )

if "FEMIDA_OCR_UPDATES_BORDER_TOP" not in os.environ:
    BORDER_UPDATES_TOP = np.array([3235, 3359, 3479, 3600, 3719, 3841])
else:
    BORDER_UPDATES_TOP = np.array(
        list(map(int, os.environ["FEMIDA_OCR_UPDATES_BORDER_TOP"].split(",")))
    )

if "FEMIDA_OCR_VARIANT_BORDER_TOP" not in os.environ:
    BORDER_VARIANT_TOP = np.array([958])
else:
    BORDER_VARIANT_TOP = np.array(
        list(map(int, os.environ["FEMIDA_OCR_VARIANT_BORDER_TOP"].split(",")))
    )

MARGIN_VERTICAL = int(os.environ.get("FEMIDA_OCR_MARGIN_VERTICAL", 83))

BORDER_BOTTOM = BORDER_TOP + MARGIN_VERTICAL
BORDER_UPDATES_BOTTOM = BORDER_UPDATES_TOP + MARGIN_VERTICAL
BORDER_VARIANT_BOTTOM = BORDER_VARIANT_TOP + MARGIN_VERTICAL

TOP_BLACK_LINE_POSITIONS = (1225, 1275)
TOP_BLACK_LINE_LEFT_RIGHT = 400
TOP_BLACK_LINE_DIFF_THRESHOLD = 40


def need_flip(image):
    # there is a black line above, we use them to guess if we need to flip image
    diff = image[  # take hand specified slides
        TOP_BLACK_LINE_POSITIONS[0] : TOP_BLACK_LINE_POSITIONS[1],
        TOP_BLACK_LINE_LEFT_RIGHT:-TOP_BLACK_LINE_LEFT_RIGHT,
    ].astype(
        float
    ) - image[  # abd from the mirrored bottom position
        -TOP_BLACK_LINE_POSITIONS[1] : -TOP_BLACK_LINE_POSITIONS[0],
        TOP_BLACK_LINE_LEFT_RIGHT:-TOP_BLACK_LINE_LEFT_RIGHT,
    ].astype(
        float
    )
    diff = diff[abs(diff) > TOP_BLACK_LINE_DIFF_THRESHOLD]
    return np.sign((diff > 0) - 0.5).mean() > 0


LABELS = ("A", "B", "C", "D", "E")
LABELS_UPDATES = ("FIRST", "SECOND", "A", "B", "C", "D", "E")
VARIANT_DIGITS = tuple(range(53, 59))
QUESTIONS = tuple(range(1, 41))
UPDATES = tuple(range(41, 53))
WIDTH = 3000
HEIGHT = int(WIDTH * 578 / 403)
Box = collections.namedtuple("Box", "center,delta,angle")


def box_to_slice(box):
    box = cv2.boxPoints(box)
    box = np.int0(box)
    xmin = min(box, key=lambda x: x[0])[0]
    xmax = max(box, key=lambda x: x[0])[0]
    ymin = min(box, key=lambda x: x[1])[1]
    ymax = max(box, key=lambda x: x[1])[1]
    return slice(ymin, ymax), slice(xmin, xmax)


@functools.lru_cache(1)
def _get_small_rectangles_positions_middle():
    xl, yt = np.meshgrid(BORDER_LEFT, BORDER_TOP)
    xr, yb = np.meshgrid(BORDER_RIGHT, BORDER_BOTTOM)
    xc, yc = (xl + xr) / 2, (yt + yb) / 2
    dx, dy = (xl - xr), (yb - yt)
    labels = LABELS
    result = []
    for i, j in itertools.product(range(xc.shape[0]), range(xc.shape[1])):
        question = QUESTIONS[j + (len(QUESTIONS) // 2) * (i // len(LABELS))]
        label = labels[i % len(LABELS)]
        box = (question, label), Box((xc[i, j], yc[i, j]), (dx[i, j], dy[i, j]), 0.0)
        result.append(box)

    xl, yt = np.meshgrid(BORDER_UPDATES_LEFT, BORDER_UPDATES_TOP)
    xr, yb = np.meshgrid(BORDER_UPDATES_RIGHT, BORDER_UPDATES_BOTTOM)
    xc, yc = (xl + xr) / 2, (yt + yb) / 2
    dx, dy = (xl - xr), (yb - yt)
    labels = LABELS_UPDATES
    # horizontal checking for updates (first number, second number, 5 answer choices)
    for i, j in itertools.product(range(xc.shape[0]), range(xc.shape[1])):
        # add (xc.shape[0]) if works with the right side of updates
        mistake = UPDATES[i + (j >= xc.shape[1] // 2) * (xc.shape[0])]
        label = labels[j % len(LABELS_UPDATES)]
        box = (mistake, label), Box((xc[i, j], yc[i, j]), (dx[i, j], dy[i, j]), 0.0)
        result.append(box)

    xl, yt = np.meshgrid(BORDER_VARIANT_LEFT, BORDER_VARIANT_TOP)
    xr, yb = np.meshgrid(BORDER_VARIANT_RIGHT, BORDER_VARIANT_BOTTOM)
    xc, yc = (xl + xr) / 2, (yt + yb) / 2
    dx, dy = (xl - xr), (yb - yt)
    # horizontal checking for variant
    for i, j in itertools.product(range(xc.shape[0]), range(xc.shape[1])):
        # only one row
        variant = VARIANT_DIGITS[j]
        label = 0
        box = (variant, label), Box((xc[i, j], yc[i, j]), (dx[i, j], dy[i, j]), 0.0)
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
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        1
    ]

    allowed_boxes = []
    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # no noise condition
        SUPERPARAM = 60  # TODO KILL feriat
        if (rect[1][0] > SUPERPARAM) and (rect[1][1] > SUPERPARAM):
            statistics = get_statistics(image, box)
            if np.mean(statistics) < 100:
                allowed_boxes.append(rect)

    try:
        nec_boxes = [
            min(allowed_boxes, key=lambda x: x[0][0] + x[0][1]),
            max(allowed_boxes, key=lambda x: x[0][0] + x[0][1]),
            min(allowed_boxes, key=lambda x: x[0][0] - x[0][1]),
            max(allowed_boxes, key=lambda x: x[0][0] - x[0][1]),
        ]
    except ValueError as e:
        raise cv2.error("No rectangles found") from e

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
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        1
    ]

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
                (rect[0][1] > ylim[0])
                and (rect[0][1] < ylim[1])
                and (rect[0][1] > xlim[0])
                and (rect[0][1] < xlim[1])
            ):
                rect = listit(rect)
                rect[1][0] += 15
                rect[1][1] += 15
                allowed_boxes.append(tuple(rect))

    return allowed_boxes


def validate_qr_code(image):
    from pyzbar import pyzbar

    barcodes = pyzbar.decode(image)
    if barcodes:
        return json.loads(barcodes[0].data.decode("utf-8"))
    else:
        return {}


class CroppedAnswers(object):
    def __init__(self, cropped, validate_qr=False):
        if need_flip(cropped):
            cropped = cropped[::-1, ::-1]
            self.flipped = True
        else:
            self.flipped = False
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
        return get_small_rectangles(self.cropped, ylim=(3000, np.inf))

    def recognized_rectangles_image(self):
        image = self.cropped.copy()
        for _, box_spec in _get_small_rectangles_positions_middle():
            box = cv2.boxPoints(box_spec)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 0, 255), 4)
        return image

    def get_rectangles_array(self, resize=None):
        res = []
        img = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2RGB)
        for box in RECTANGLES_POSITIONS_MIDDLE.values():
            if resize is not None:
                res.append(cv2.resize(img[box_to_slice(box)], (resize, resize)))
            else:
                res.append(img[box_to_slice(box)])
        return (np.stack(res).transpose((0, 3, 1, 2)) / 255.0).astype(np.float32)

    def get_rectangles_with_labels(self):
        return self.get_rectangles_array(), list(RECTANGLES_POSITIONS_MIDDLE.keys())

    @staticmethod
    def get_labels():
        return RECTANGLES_POSITIONS_MIDDLE.keys()

    def parse_qr(self):
        return validate_qr_code(self.cropped)

    def plot_predicted(self, labels, only_answers=False):
        recognized = self.cropped.copy()
        for p, (label, box) in zip(labels, RECTANGLES_POSITIONS_MIDDLE.items()):
            if p:
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(recognized, [box], -1, (0, 0, 255), 4)
        if only_answers:
            recognized = recognized[self.ANSWERS_BOX]
        return recognized

    PERSONAL_BOX = (slice(20, 1100), slice(45, 3000))
    ANSWERS_BOX = (slice(1150, -200), slice(45, 3000))
    MATH_CHECKBOX = (slice(780, 870), slice(490, 570))
    OT_CHECKBOX = (slice(780, 870), slice(1220, 1300))

    @property
    def personal(self):
        return self.cropped[self.PERSONAL_BOX]

    @property
    def answers(self):
        return self.cropped[self.ANSWERS_BOX]

    @property
    def math_checkbox(self):
        return self.cropped[self.MATH_CHECKBOX]

    @property
    def ot_checkbox(self):
        return self.cropped[self.OT_CHECKBOX]
