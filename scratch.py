from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda, Dropout
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from matplotlib.pyplot import imshow
from keras.models import Model
import numpy as np
from shutil import copy2
import pandas as pd
import os

input_tensor = Input(shape=(224, 224, 3))
tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)

base_xception = Xception(input_tensor=input_tensor, weights="imagenet", include_top=True)
xception = Model(input=input_tensor, output=base_xception(tf_inputs))

