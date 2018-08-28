import numpy as np
import os
import tempfile
import pytest
import cv2
import femida_detect.pdf2img


def test_pdf_to_image(pdf):
    image = next(femida_detect.pdf2img.pdf_to_images(pdf))
    assert isinstance(image, np.ndarray)
    pytest.helpers.save_image(image, 'test_pdf_to_image', os.path.basename(pdf)+'.jpg')


def test_pdf_to_image_silent(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'img%03d.png')
        fpath = femida_detect.pdf2img.pdf_to_images_silent(pdf, out)[0]
        assert isinstance(cv2.imread(fpath), np.ndarray)