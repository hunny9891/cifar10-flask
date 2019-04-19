import os
import tensorflow as tf
from keras.models import load_model as lm

# Constants
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
IMAGES_PATH = os.path.join(ROOT_DIR, 'images')
MODEL_NAME = 'custom_resnet50.h5'

LABEL_DICT = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def getModelPath():
    # Test whether the directory exists
    if not os.path.isdir(os.path.join(MODEL_PATH)):
        print("Directory: " + MODEL_PATH +
              " not found. Creating directory structure.")
        os.makedirs(MODEL_PATH)
    filepath = os.path.join(MODEL_PATH, MODEL_NAME)
    return filepath


def init():
    print("Loading model...")
    model = lm(getModelPath())
    print("Model loaded successfully.")
    graph = tf.get_default_graph()

    return graph, model
