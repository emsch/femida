import pytest
import os
import cv2


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
            '..', 'results', 'tests', *to[:-1])
    os.makedirs(dest, exist_ok=True)
    cv2.imwrite(
        os.path.join(dest, to[-1]),
        array
    )
