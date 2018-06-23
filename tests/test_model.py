import torch
import pytest
import itertools
from femida_detect.detect import select


@pytest.mark.parametrize(
    ['size', 'v'],
    itertools.product([20, 28, 30], ['v1', 'v2', 'v3'])
)
def test_model(size, v):
    inp = torch.rand(10, 3, size, size)
    out = select[v](3, size)(inp)
    assert out.shape == (10, 1)
