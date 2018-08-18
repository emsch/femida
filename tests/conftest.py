import pytest
import os


@pytest.fixture('session', params=['smallpdf.pdf'])
def pdf(request):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        request.param
    )
