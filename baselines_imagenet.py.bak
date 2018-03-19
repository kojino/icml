from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
import os
import logging as log
import numpy as np
from noise_functions_dl import GradientDescentDL, gradientDescentFunc
from functools import partial
from mwu import adversary
import tensorflow as tf
import argparse
import sys
from keras.models import Model
from keras.layers import Average


def ensembleModels(models, model_input):
    # taken from https://medium.com/@twt446/ensemble-and-store-models-in-keras-2-x-b881a6d7693f
    yModels=[model(model_input) for model in models]
    yAvg=Average()(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    return modelEns


def main(arguments):

    parser = argparse.ArgumentParser(description="deep learning classification experiments argument parser")
    parser.add_argument("-noise_type", choices=["targeted", "untargeted"], type=str, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-opt_iters", help="number of iterations to run optimizer", type=int, required=True)
    parser.add_argument("-learning_rate", help="learning rate for the optimizer", type=float, required=True)
    parser.add_argument("-model", help="number of model to choose", type=int, required=True)
    args = parser.parse_args(arguments)

    data_path = "imagenet_data"
    exp_name = "imagenet_baseline_{}_{}".format(args.noise_type, args.model)
    log_file = exp_name + ".log"

    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(exp_name + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # add preprocessing layer for each individual model
        input_tensor = Input(shape=(224, 224, 3))
        tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
        caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)

        base_inception = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=True)
        inception = Model(inputs=input_tensor, outputs=base_inception(tf_inputs))

        base_resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=True)
        resnet = Model(inputs=input_tensor, outputs=base_resnet(caffe_inputs))

        base_inceptionresnet = InceptionResNetV2(input_tensor=input_tensor, weights="imagenet", include_top=True)
        inceptionresnet = Model(inputs=input_tensor, outputs=base_inceptionresnet(tf_inputs))

        base_vgg = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=True)
        vgg = Model(inputs=input_tensor, outputs=base_vgg(caffe_inputs))

        models = [vgg, inceptionresnet, resnet, inception]

        for model in models:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model_input = Input(shape=models[0].input_shape[1:])
        ensemble = ensembleModels(models, model_input)
        ensemble.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        X_exp = np.load(data_path + "/" + "X_exp.npy")
        Y_exp = np.load(data_path + "/" + "Y_exp.npy")
        Target_exp = np.load(data_path + "/" + "Target_exp.npy")

        data_dims = (224, 3, 1000)
        box_vals = (0.0, 255.0)
        X_exp = X_exp[:50]
        Y_exp = Y_exp[:50]
        Target_exp = Target_exp[:50]

        log.debug("Ensemble Accuracy {}".format(ensemble.evaluate(X_exp, Y_exp)))

        target_bool = args.noise_type == "targeted"
        print "TARGET BOOL", target_bool
        if args.model < 4:
            exp_models = [models[args.model]]
        else:
            exp_models = [ensemble]

        attack_obj = GradientDescentDL(sess, exp_models, args.alpha, data_dims, box_vals, targeted=target_bool,
                                       batch_size=1, max_iterations=args.opt_iters,
                                       learning_rate=args.learning_rate, confidence=0)

        noise_func = partial(gradientDescentFunc, attack=attack_obj)
        targeted = Target_exp if target_bool else False

        noise = adversary(np.array([1.0]), [ensemble], X_exp, Y_exp, args.alpha, noise_func, targeted)
        np.save(exp_name + "/noise.npy", noise)

        log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])