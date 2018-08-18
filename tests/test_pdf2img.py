import numpy as np
import femida_detect.pdf2img
import os
import pytest


def test_pdf_to_image(pdf):
    image = femida_detect.pdf2img.pdf_to_images(pdf)[0]
    assert isinstance(image, np.ndarray)
    pytest.helpers.save_image(image, 'test_pdf_to_image', os.path.basename(pdf)+'.jpg')
