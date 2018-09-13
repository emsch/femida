import pytest
import os
import cv2
from femida_detect.imgparse import (
    crop_image,
    CroppedAnswers
)


def test_crop_image(jpg):
    cropped = crop_image(cv2.imread(jpg))
    pytest.helpers.save_image(cropped, 'test_crop_image', os.path.basename(jpg) + '.cropped.jpg')


def test_recognized_positions(cropped):
    recognized = CroppedAnswers.from_file(cropped).recognized_rectangles_image()
    pytest.helpers.save_image(recognized, 'test_recognized_positions', os.path.basename(cropped) + '.recognized.jpg')
