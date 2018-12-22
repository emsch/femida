import pytest
import os
import cv2
from femida_detect.imgparse import (
    crop_image,
    CroppedAnswers,
    need_flip
)


def test_crop_image(jpg):
    cropped = crop_image(cv2.imread(jpg))
    pytest.helpers.save_image(cropped, 'test_crop_image', os.path.basename(jpg) + '.cropped.jpg')


def test_recognized_positions(cropped):
    recognized = CroppedAnswers.from_file(cropped).recognized_rectangles_image()
    pytest.helpers.save_image(recognized, 'test_recognized_positions', os.path.basename(cropped) + '.recognized.jpg')


def test_flipping(jpg):
    cropped = crop_image(cv2.imread(jpg))
    assert not need_flip(cropped)
    assert need_flip(cropped[::-1, ::-1])
    cropped_answers = CroppedAnswers(cropped[::-1, ::-1])
    assert not need_flip(cropped_answers.cropped)
    assert 0
