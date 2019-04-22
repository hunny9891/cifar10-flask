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


def readAndNormalizeImg():
    images = []
    print("Reading images from directory...")
    for img_path in os.listdir(util.IMAGES_PATH):
        img = image.load_img(os.path.join(
            util.IMAGES_PATH, img_path), target_size=(32, 32, 3))
        images.append(image.img_to_array(img))

    # Convert to numpy array
    X = np.asarray(images, dtype=np.float)
    X /= 255

    # Subtract mean pixel
    mean = np.mean(X, axis=0)
    X -= mean

    return images, X

# Unit Test


def main():
    images, X = readAndNormalizeImg()
    preds = predict(X)
    for pred in preds:
        print(util.LABEL_DICT[pred])
