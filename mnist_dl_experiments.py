import logging as log
import argparse
from mwu import runMWU
import sys
import datetime
from setup_mnist import *
from functools import partial
import numpy as np
import tensorflow as tf
import os
from noise_functions_dl import GradientDescentDL, gradientDescentFunc


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
    exp_name = "deepLearning-{}-{}-{}-{}-{}{}".format(args.data_path, args.noise_type, date.month, date.day,
                                                      date.hour, date.minute)
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

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        log.debug("\nbeginning to load models...")

        model_dir = "deep_networks"
        models = [conv_net(1, model_dir + "/conv1"), conv_net(0, model_dir + "/conv2"),
                  multilayer(4, 128, model_dir + "/mlp1"), multilayer(2, 256, model_dir + "/mlp2"),
                  multilayer(0, 0, model_dir + "/zero_layer")]

        for model in models:
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        log.debug("finished loading models!\n")

        X_exp = np.load(args.data_path + "/" + "X_exp.npy")
        Y_exp = np.load(args.data_path + "/" + "Y_exp.npy")
        Target_exp = np.load(args.data_path + "/" + "Target_exp.npy")

        X_exp = X_exp.reshape(-1, 28, 28, 1)
        Y_exp = np.array([(np.arange(10) == l).astype(np.float32) for l in Y_exp])
        Target_exp = np.array([(np.arange(10) == l).astype(np.float32) for l in Target_exp])

        log.debug("Num Points {}\n".format(X_exp.shape[0]))
        target_bool = args.noise_type == "targeted"
        # initialize the attack object_
        attack_obj = GradientDescentDL(sess, models, args.alpha, (28, 1, 10), -.5, .5, # TODO: add these paramaters as variables
                                       targeted=target_bool, batch_size=1, max_iterations=args.opt_iters,
                                       learning_rate=args.learning_rate, confidence=0)

        log.debug("starting attack!")
        noise_func = partial(gradientDescentFunc, attack=attack_obj)
        targeted = Target_exp if target_bool else False
        weights, noise, loss_history, acc_history, action_loss = runMWU(models, args.mwu_iters, X_exp, Y_exp, args.alpha,
                                                                        noise_func, exp_name, targeted=targeted,
                                                                        dl=True)

        np.save(exp_name + "/" + "weights.npy", weights)
        np.save(exp_name + "/" + "noise.npy", noise)
        np.save(exp_name + "/" + "loss_history.npy", loss_history)
        np.save(exp_name + "/" + "acc_history.npy", acc_history)
        np.save(exp_name + "/" + "action_loss.npy", action_loss)

        log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
