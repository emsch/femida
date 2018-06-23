import torch
import pytest
from femida_detect.detect import Modelv1


@pytest.mark.parametrize(
    [20, 28, 30]
)
def test_modelv1(size):
    inp = torch.rand(10, 3, size, size)
    out = Modelv1(3, size)(inp)
    assert out.shape == (10, 2)
