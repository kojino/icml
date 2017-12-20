import logging as log
import argparse
from setup_mnist import MNIST
import os
import helper
import numpy as np
from binary_models import trainLBC
from noise_functions_binary import FUNCTION_DICT
from mwu import runMWU
import sys


def main(arguments):
    parser = argparse.ArgumentParser(description="binary classification experiments argument parser")

    parser.add_argument("-classes", help="list of two integers in [10] to use for classification", nargs="+",
                        type=int, required=True)
    parser.add_argument("-noise_func", help="noise function used for the adversary",
                        choices=["randomAscent", "greedyAscent", "binaryOracle", "gradientDescent"], required=True)
    parser.add_argument("-iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-num_classifiers", help="number of classifiers to use", type=int, required=True)
    parser.add_argument("-exp_dir", help="name of experiments", type=str, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-num_points", help="number of points to include in experiment", type=int, required=True)
    parser.add_argument("-log_file", help="name of the log file", type=str, required=True)

    args = parser.parse_args(arguments)

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(args.exp_dir + "/" + args.log_file)
    log.getLogger().addHandler(file_handler)

    log.debug("BINARY CLASSIFICATION: \n")
    log.debug("Noise Function {}".format(args.noise_func))
    log.debug("Iters {} ".format(args.iters))
    log.debug("Classes {} ".format(args.classes))
    log.debug("Num Classifiers {}".format(args.num_classifiers))
    log.debug("Alpha {}".format(args.alpha))
    log.debug("Num Points {}\n".format(args.num_points))

    data = MNIST()
    label_dict = dict(zip(args.classes, [-1, 1]))
    X_train, Y_train = helper.subsetData(data.train_data, data.train_labels, label_dict)
    X_train, Y_train, permutation = helper.shuffleArraysInUnison(X_train, Y_train)

    np.save(args.exp_dir + "/" + "permutation.npy", permutation)

    X_test, Y_test = helper.subsetData(data.test_data, data.test_labels, label_dict)

    train_size = len(X_train) / args.num_classifiers

    models = []

    log.debug("training linear, binary classifiers")
    for i in xrange(args.num_classifiers):
        start = train_size * i
        end = start + train_size
        model = trainLBC(X_train[start:end], Y_train[start:end])
        log.debug("Model {}, Start {}, End {}, Test Accuracy {}".format(i, start, end, model.evaluate(X_test, Y_test)))
        models.append(model)

    log.debug("selecting {} points correctly classified by all models".format(args.num_points))

    X_exp, Y_exp = helper.generate_data(args.num_points, X_test, Y_test, models)

    np.save(args.exp_dir + "/" + "X_exp.npy", X_exp)
    np.save(args.exp_dir + "/" + "Y_exp.npy", Y_exp)

    noise_func = FUNCTION_DICT[args.noise_func]

    weights, noise, loss_history, max_acc_history, action_loss = runMWU(models, args.iters, X_exp, Y_exp, args.alpha,
                                                                        noise_func)



    np.save(args.exp_dir + "/" + "weights.npy", weights)
    np.save(args.exp_dir + "/" + "noise.npy", noise)
    np.save(args.exp_dir + "/" + "loss_history.npy", loss_history)
    np.save(args.exp_dir + "/" + "max_acc_history.npy", max_acc_history)
    np.save(args.exp_dir + "/" + "action_loss.npy", action_loss)

    log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
