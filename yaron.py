import logging as log
import numpy as np
from setup_mnist import *
from keras.models import Input, Model
from keras.layers import Average
import tensorflow as tf
from noise_functions_dl import GradientDescentDL, gradientDescentFunc
from functools import partial
from mwu import adversary
import time

data = MNIST()
X_exp = np.load("multiclass_data_2/X_exp.npy")
Y_exp = np.load("multiclass_data_2/Y_exp.npy")
Target_exp = np.load("multiclass_data_2/Target_exp.npy")


def ensembleModels(models, model_input):
    # taken from https://medium.com/@twt446/ensemble-and-store-models-in-keras-2-x-b881a6d7693f
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models]
    # averaging outputs
    yAvg=Average()(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    return modelEns


def test_accuracy(V):
    model_dir = "deep_networks"
    models = [conv_net(False, 2, 200, model_dir + "/conv0"), conv_net(True, 2, 200, model_dir + "/conv1"),
              conv_net(True, 4, 64, model_dir + "/conv2"), multilayer(4, 128, model_dir + "/mlp0"),
              multilayer(2, 256, model_dir + "/mlp1")]
    return max([model.evaluate(X_exp + V)[1] for model in models])


def generate_ensemble_noise(alpha):
    lr = .001
    opt_iters = 3000

    with tf.Session() as sess:
        model_dir = "deep_networks"
        models = [conv_net(False, 2, 200, model_dir + "/conv0"), conv_net(True, 2, 200, model_dir + "/conv1"),
                  conv_net(True, 4, 64, model_dir + "/conv2"), multilayer(4, 128, model_dir + "/mlp0"),
                  multilayer(2, 256, model_dir + "/mlp1")]
        model_input = Input(shape=models[0].input_shape[1:])
        ensemble = ensembleModels(models, model_input)
        ensemble.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        attack_obj = GradientDescentDL(sess, [ensemble], alpha, (28, 1, 10), (-.5, .5),
                                       targeted=False, batch_size=1, max_iterations=opt_iters,
                                       learning_rate=lr, confidence=0)
        noise_func = partial(gradientDescentFunc, attack=attack_obj)
        return adversary(np.array([1.0]), [ensemble], X_exp, Y_exp, alpha, noise_func, False)


if __name__ == "__main__":

    exp_name = "yaron_noise_comparison"
    log_file = exp_name + ".log"
    target_acc = .4678
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    file_handler = log.FileHandler(exp_name + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    alpha = 4.0
    start = time.time()

    while True:
        log.debug("Current Alpha {}".format(alpha))

        noise = generate_ensemble_noise(alpha)
        res = test_accuracy(noise)

        log.debug("Current Max Accuracy {} ".format(res))

        if res <= target_acc:
            log.debug("Found Successful Alpha")
            np.save(exp_name + "/" + "noise.npy", noise)
            log.debug("Final Alpha {} ".format(alpha))
            break

        diff = time.time() - start
        start = time.time()
        log.debug("Time in Search {}".format(diff))
        alpha += .1

    log.debug("Success")
