import pytest
import os
import cv2
import torch
from femida_detect.detect import select
from femida_detect.imgparse import (
    crop_image, CroppedAnswers
)

pytest_plugins = ['helpers_namespace']


@pytest.fixture('session', params=['smallpdf.pdf'])
def pdf(request):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        request.param
    )


@pytest.fixture('session', params=['variant.jpg'])
def jpg(request):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        request.param
    )


@pytest.helpers.register
def save_image(array, *to):
    assert len(to) >= 1
    dest = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'out', *to[:-1])
    os.makedirs(dest, exist_ok=True)
    cv2.imwrite(
        os.path.join(dest, to[-1]),
        array
    )


@pytest.fixture('module')
def cropped(jpg):
    return crop_image(cv2.imread(jpg))


@pytest.fixture('module')
def cropped_answers(cropped):
    return CroppedAnswers(cropped)


@pytest.fixture('module')
def model():
    net = select['v3'](3, 28)
    for p in net.parameters():
        p.requires_grad = False
    return net.eval()
