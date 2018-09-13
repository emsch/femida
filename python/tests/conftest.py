import pytest
import os
import cv2
from femida_detect.detect import select, eval
from femida_detect.imgparse import (
    CroppedAnswers
)

pytest_plugins = ['helpers_namespace']
INPUT = ['newM.pdf', 'newOT.pdf']


@pytest.fixture('session', params=INPUT)
def pdf(request):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        request.param
    )


@pytest.fixture('session')
def jpg(pdf):
    return pdf + '.jpg'


@pytest.fixture('session')
def cropped(jpg):
    return jpg + '.cropped.jpg'


@pytest.fixture('session')
def cropped_answers(cropped):
    return CroppedAnswers.from_file(cropped)


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


@pytest.fixture('session')
def model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'model',
        'model.t7'
    )
    if os.path.exists(model_path):
        net = eval.load_net(model_path)
    else:
        import warnings
        warnings.warn('model not found, using random initialization')
        net = select['v3'](3, 28)
        for p in net.parameters():
            p.requires_grad = False
    return net.eval()
