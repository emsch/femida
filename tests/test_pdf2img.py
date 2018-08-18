from PIL.Image import Image
import femida_detect.pdf2img


def test_pdf_to_image(pdf):
    image = femida_detect.pdf2img.pdf_to_images(pdf)[0]
    assert isinstance(image, Image)
