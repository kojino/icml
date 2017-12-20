import numpy as np
import matplotlib.pyplot as plt
from noise_functions_binary import tryRegionBinary


def findNoiseBounds(models, X, Y):
    max_bounds = []
    num_models = len(models)
    for i in xrange(len(X)):
        max_r = -1 * Y[i] * np.ones(num_models)
        max_v = tryRegionBinary(models, max_r, X[i])
        max_bounds.append(np.linalg.norm(max_v))
    min_bounds = np.array([model.distance(X) for model in models]).T
    min_bounds = np.mean(min_bounds, axis=1)
    return max_bounds, min_bounds


def shuffleArraysInUnison(a, b, p=None):
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    if p is None:
        p = np.random.permutation(len(a))
    return a[p], b[p], p


def generate_data(num_pts, X, Y, models, target_dict=False):
    num_selected = 0
    num_models = len(models)
    resX = []
    resY = []
    for i in xrange(len(X)):
        all_correct = sum([model.evaluate(X[i:i+1], Y[i:i+1]) for model in models]) == num_models
        if all_correct:
            if target_dict:
                true_label = np.argmax(Y[i])
                target_labels = target_dict[true_label]
                for l in target_labels:
                    resX.append(X[i])
                    resY.append((np.arange(10) == l).astype(np.float32))
            else:
                resX.append(X[i])
                resY.append(Y[i])
            num_selected += 1
        if num_selected == num_pts:
            break
    if num_selected < num_pts:
        print "Not enough points were correctly predicted by all models"
    return np.array(resX), np.array(resY)


def subsetData(data, labels, label_dict):
    """
    used for binary classification TODO: fill out comments
    :param data:
    :param labels:
    :param label_dict:
    :return:
    """
    subset = set(label_dict.keys())
    X = []
    Y = []
    for i in xrange(len(data)):
        label = np.argmax(labels[i])
        if label in subset:
            label = label_dict[label]
            X.append(data[i].reshape(-1,))
            Y.append(label)
    return np.array(X), np.array(Y)


def showImage(img):
    dim = int(np.sqrt(img.shape[0]))
    plt.imshow(img.reshape(dim, dim), cmap='gray')