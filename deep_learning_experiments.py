import logging as log
import argparse
from mwu import runMWU
import sys
import datetime
from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf
import os
from noise_functions_dl import GradientDescentDL

def gradientDescent()



def main(arguments):
    parser = argparse.ArgumentParser(description="deep leanrning classification experiments argument parser")
    parser.add_argument("-noise_type", help="targeted or untargeted noise", choices=["targeted", "untargeted"],
                        default="untargeted", type=str)
    parser.add_argument("-mwu_iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-data_path", help="directory with experiment data + models", type=str, required=True)
    parser.add_argument("-opt_iters", help="number of iterations to run optimizer", type=int, required=True)
    parser.add_argument("-learning_rate", help="learning rate for the optimizer", type=float, required=True)
    args = parser.parse_args(arguments)

    date = datetime.datetime.now()
    exp_name = "deepLearning-{}-{}-{}-{}{}".format(args.noise_type, date.month, date.day, date.hour, date.minute)
    log_file = exp_name + ".log"

    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    file_handler = log.FileHandler(exp_name + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    log.debug("Noise Type {}".format(args.noise_type))
    log.debug("MWU Iters {} ".format(args.mwu_iters))
    log.debug("Alpha {}".format(args.alpha))
    log.debug("Learning Rate {}".format(args.learning_rate))
    log.debug("Optimization Iters {}".format(args.opt_iters))
    log.debug("Data path : {}".format(args.data_path))

    # base_xception = Xception(input_tensor=input_tensor, weights="imagenet", include_top=True)
    # xception = Model(input=input_tensor, output=base_xception(tf_inputs))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
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

        models = [inception, resnet, inceptionresnet, vgg]

        X_exp = np.load(args.data_path + "/" + "X_exp.npy")

        if args.noise_type == "targeted":
            Y_exp = np.load(args.data_path + "/" + "Y_true_exp.npy")
        else:
            Y_exp = np.load(args.data_path + "/" + "Y_target_exp.npy")

        log.debug("Num Points {}\n".format(X_exp.shape[0]))

        targeted = args.noise_type == "targeted"
        noise_func = GradientDescentDL(sess, models, args.alpha, targeted=targeted, batch_size=1,
                                       max_iterations=args.opt_iters, learning_rate=args.learning_rate, confidence=0)

        noise_func.attack(X_exp, Y_exp, np.ones(len(models)))

        # weights, noise, loss_history, acc_history, action_loss = runMWU(models, args.iters, X_exp, Y_exp, args.alpha,
        #                                                                 noise_func, exp_name,)

        # np.save(exp_name + "/" + "weights.npy", weights)
        # np.save(exp_name + "/" + "noise.npy", noise)
        # np.save(exp_name + "/" + "loss_history.npy", loss_history)
        # np.save(exp_name + "/" + "acc_history.npy", acc_history)
        # np.save(exp_name + "/" + "action_loss.npy", action_loss)

        log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
