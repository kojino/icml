from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input
import argparse
import logging as log
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
from l2_attack import CarliniL2
import tensorflow as tf
import os
import sys


def main(arguments):
    parser = argparse.ArgumentParser(description="dawn song comparison argument parser")

    parser.add_argument("-exp_dir", help="name of experiments", type=str, required=True)
    parser.add_argument("-log_file", help="name of the log file", type=str, required=True)
    parser.add_argument("-targeted", help="boolean value to determine character of the noise", choices=[0, 1],
                        type=int, required=True)
    parser.add_argument("-data_path", help="directory with experiment data + models", type=str, required=True)

    args = parser.parse_args(arguments)

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(args.exp_dir + "/" + args.log_file)
    log.getLogger().addHandler(file_handler)

    # config=tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # create the models
        input_tensor = Input(shape=(224, 224, 3))
        tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
        caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)

        #base_xception = Xception(input_tensor=input_tensor, weights="imagenet", include_top=True)
        #xception = Model(input=input_tensor, output=base_xception(tf_inputs))

        base_inception = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=True)
        inception = Model(input=input_tensor, output=base_inception(tf_inputs))

        base_resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=True)
        resnet = Model(input=input_tensor, output=base_resnet(caffe_inputs))
        #
        base_inceptionresnet = InceptionResNetV2(input_tensor=input_tensor, weights="imagenet", include_top=True)
        inceptionresnet = Model(input=input_tensor, output=base_inceptionresnet(tf_inputs))
        #
        base_vgg = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=True)
        vgg = Model(input=input_tensor, output=base_vgg(caffe_inputs))

        #models = [xception]

        models = [inception, resnet, inceptionresnet, vgg]

        for model in models:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        X_exp = np.load(args.data_path + "/" + "X_exp.npy")[:3]
        Y_true_exp = np.load(args.data_path + "/" + "Y_true_exp.npy")[:3]
        Y_target_exp = np.load(args.data_path + "/" + "Y_target_exp.npy")

        print "True Labels ", np.argmax(Y_true_exp, axis=1)

        num_models = len(models)
        weights = [1.0 / num_models] * num_models
        print "Starting attack"
        carlini = CarliniL2(sess, models, targeted=args.targeted, batch_size=1, max_iterations=9000,
                            binary_search_steps=9, confidence=0)

        untargeted_adv = carlini.attack(X_exp, Y_true_exp, weights)

        untargeted_results = []

        for i in xrange(len(untargeted_adv)):
            true_label = np.argmax(Y_true_exp[i])

            distortion = np.sum((untargeted_adv[i] - X_exp[i]) ** 2) ** .5

            predicted_labels = [np.argmax(model.predict(untargeted_adv[i:i + 1])) for model in models]
            success_vector = (np.array(predicted_labels) != true_label).astype(np.float32)

            res = [i, true_label, distortion] + predicted_labels + list(success_vector)
            untargeted_results.append(res)
            #
            # imsave(exp_dir + "/images/untargeted/" + str(i) + "_" + str(true_label) + "_untargeted.jpg",
            #        untargeted_adv[i].reshape(28, 28))

        np.save(args.exp_dir + "/untargeted_results.npy", np.array(untargeted_results))

        print "UnTargeted Results"
        for row in untargeted_results:
            print row


if __name__ =="__main__":
    main(sys.argv[1:])






