from __future__ import annotations
from typing import Any
import math

import numpy as np
from scipy import ndimage
from nptyping import NDArray, Number
import pickle
from PIL import Image
import cv2



def save_model(model: Sequential, filename: str) -> bool:
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return True


def load_model(filename: str) -> Sequential:
    with open(filename, 'rb') as f:
        pre_trained_model = pickle.load(f)
    return pre_trained_model


"""
The following code is retrieved and modified from tensorflow-mnist
https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
https://github.com/opensourcesblog/tensorflow-mnist
"""

def preprocessing(img: Image.Image) -> NDArray[Any, Number]:
    img = img.convert('1')
    img_arr = np.asarray(img, dtype=np.float_)

    if img_arr.max() == 0:
        return -1

    while img_arr[0, :].max() == 0:
        img_arr = img_arr[1:, :]

    while img_arr[-1, :].max() == 0:
        img_arr = img_arr[:-1, :]

    while img_arr[:, 0].max() == 0:
        img_arr = img_arr[:, 1:] 

    while img_arr[:, -1].max() == 0:
        img_arr = img_arr[:, :-1] 

    rows,cols = img_arr.shape    

    if rows > cols:
        factor = 20 / rows
        rows = 20
        cols = int(round(cols * factor))
        img_arr = cv2.resize(img_arr, (cols, rows))
    else:
        factor = 20./ cols
        cols = 20
        rows = int(round(rows * factor))
        img_arr = cv2.resize(img_arr, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2)), int(math.floor((28 - cols) / 2)))
    rowsPadding = (int(math.ceil((28 - rows) / 2)), int(math.floor((28 - rows) / 2)))
    img_arr = np.lib.pad(img_arr, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(img_arr)
    shifted = shift(img_arr, shiftx, shifty)

    return cv2.resize(shifted, (32, 32))


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx],[0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted