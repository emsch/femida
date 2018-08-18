import pytest
import os


@pytest.fixture('session')
def smallpdf():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'smallpdf.pdf'
    )
