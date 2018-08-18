import pytest
from PIL.Image import Image
import femida_detect.pdf2img


def test_pdf_to_image(smallpdf):
    image = femida_detect.pdf2img.pdf_to_image(smallpdf)
    assert isinstance(image, Image)
