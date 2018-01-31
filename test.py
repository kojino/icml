from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np


input_tensor = Input(shape=(224, 224, 3))
tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)

base_inception = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=True)
inception = Model(input=input_tensor, output=base_inception(tf_inputs))

base_resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=True)
resnet = Model(input=input_tensor, output=base_resnet(caffe_inputs))

base_inceptionresnet = InceptionResNetV2(input_tensor=input_tensor, weights="imagenet", include_top=True)
inceptionresnet = Model(input=input_tensor, output=base_inceptionresnet(tf_inputs))

base_vgg = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=True)
vgg = Model(input=input_tensor, output=base_vgg(caffe_inputs))

models = [vgg, inceptionresnet, resnet, inception]

for model in models:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


X_exp = np.load("imagenet_data/X_exp.npy")
Y_exp = np.load("imagenet_data/Y_exp.npy")
for i in xrange(len(models)):
    print models[i].evaluate(X_exp, Y_exp)