import logging as log
import argparse
import os
import numpy as np
from binary_models import LinearBinaryClassifier
from noise_functions_binary import FUNCTION_DICT
from mwu import runMWU
import sys


def main(arguments):
    parser = argparse.ArgumentParser(description="binary classification experiments argument parser")

    parser.add_argument("-noise_func", help="noise function used for the adversary",
                        choices=["randomAscent", "greedyAscent", "binaryOracle", "gradientDescent"], required=True)
    parser.add_argument("-iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-exp_dir", help="name of experiments", type=str, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-log_file", help="name of the log file", type=str, required=True)
    parser.add_argument("-data_path", help="directory with experiment data + models", type=str, required=True)
    parser.add_argument("-num_classifiers", help="number of classifiers", type=int, required=True)
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

    X_test = np.load(args.data_path + "/" + "X_test.npy")
    Y_test = np.load(args.data_path + "/" + "Y_test.npy")

    models = []

    for i in xrange(args.num_classifiers):
        weights = np.load(args.data_path + "/" + "weights_{}.npy".format(i))
        bias = np.load(args.data_path + "/" + "bias_{}.npy".format(i))
        model = LinearBinaryClassifier(weights, bias)
        log.debug("Model {}, Test Accuracy {}".format(i, model.evaluate(X_test, Y_test)))
        models.append(model)

    log.debug("Num Classifiers {}".format(args.num_classifiers))
    log.debug("Alpha {}".format(args.alpha))

    X_exp = np.load(args.data_path + "/" + "X_exp.npy")
    Y_exp = np.load(args.data_path + "/" + "Y_exp.npy")

    log.debug("Num Points {}\n".format(X_exp.shape[0]))

    noise_func = FUNCTION_DICT[args.noise_func]

    weights, noise, loss_history, max_acc_history, action_loss = runMWU(models, args.iters, X_exp, Y_exp, args.alpha,
                                                                        noise_func, args.exp_dir)

    np.save(args.exp_dir + "/" + "weights.npy", weights)
    np.save(args.exp_dir + "/" + "noise.npy", noise)
    np.save(args.exp_dir + "/" + "loss_history.npy", loss_history)
    np.save(args.exp_dir + "/" + "max_acc_history.npy", max_acc_history)
    np.save(args.exp_dir + "/" + "action_loss.npy", action_loss)

    log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
