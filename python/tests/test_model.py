import torch
import pytest
import itertools
import cv2
import numpy as np
import os
from femida_detect.detect import select
from femida_detect.imgparse import RECTANGLES_POSITIONS_MIDDLE


@pytest.mark.parametrize(
    ['size', 'v'],
    itertools.product([20, 28, 30], ['v1', 'v2', 'v3', 'v5'])
)
def test_model(size, v):
    inp = torch.rand(10, 3, size, size)
    out = select[v](3, size)(inp)
    assert out.shape == (10, 1)


def test_real_data_predict(cropped_answers, model, cropped):
    images = cropped_answers.get_rectangles_array(28)
    pred = model(torch.from_numpy(images)) < .5
    recognized = cropped_answers.cropped.copy()
    for p, (label, box) in zip(
        pred.numpy(), RECTANGLES_POSITIONS_MIDDLE.items()
    ):
        if p:
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(recognized, [box], -1, (0, 0, 255), 4)
    pytest.helpers.save_image(recognized, 'test_real_data_predict', os.path.basename(cropped) + '.inferred.jpg')
