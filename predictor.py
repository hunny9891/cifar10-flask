import os

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

import util
from util import *

global model, graph
graph, model = init()


def predict(X_Input):
    with graph.as_default():
        predictions = model.predict(X_Input)
        best_pred = np.argmax(predictions, axis=1)
        return best_pred


def readAndNormalizeImg(img):
    img = image.load_img((img), target_size=(32, 32, 3))
    img_asarr = image.img_to_array(img)

    # Convert to numpy array
    X = np.asarray(img_asarr, dtype=np.float)
    X /= 255

    # Subtract mean pixel
    mean = np.mean(X, axis=0)
    X -= mean

    # Reshape to four dimension as (1, 32,32,3)
    X = np.reshape(X, (1,32,32,3))
    
    return X

# Unit Test


def main():
    images, X = readAndNormalizeImg()
    preds = predict(X)
    for pred in preds:
        print(util.LABEL_DICT[pred])
