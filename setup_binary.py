from setup_mnist import MNIST
import helper
import os
import numpy as np
from binary_models import trainLBC

classes = [4, 9]
num_classifiers = 5
num_points = 200
folder = "binary_data"

data = MNIST()
label_dict = dict(zip(classes, [-1, 1]))

if not os.path.exists(folder):
    os.mkdir(folder)

X_train, Y_train = helper.subsetData(data.train_data, data.train_labels, label_dict)
X_train, Y_train, permutation = helper.shuffleArraysInUnison(X_train, Y_train)

np.save(folder + "/" + "permutation.npy", permutation)

X_test, Y_test = helper.subsetData(data.test_data, data.test_labels, label_dict)

train_size = len(X_train) / num_classifiers

models = []
training_sets = []
for i in xrange(num_classifiers):
    start = train_size * i
    end = start + train_size
    training_sets.append([start, end])
    model = trainLBC(X_train[start:end], Y_train[start:end])
    print("Model {}, Start {}, End {}, Test Accuracy {}".format(i, start, end, model.evaluate(X_test, Y_test)))
    np.save(folder + "/" + "weights_{}.npy".format(i), model.weights)
    np.save(folder + "/" + "bias_{}.npy".format(i), model.bias)
    models.append(model)

X_exp, Y_exp = helper.generate_data(num_points, X_test, Y_test, models)

np.save(folder + "/" + "X_exp.npy", X_exp)
np.save(folder + "/" + "Y_exp.npy", Y_exp)
np.save(folder + "/" + "X_test.npy", X_test)
np.save(folder + "/" + "Y_test.npy", Y_test)
np.save(folder + "/" + "training_sets.npy", np.array(training_sets))

