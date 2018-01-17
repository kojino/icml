from setup_mnist import MNIST
import helper
import os
import numpy as np
from linear_models import trainLMC

num_classifiers = 5
num_points = 100
folder = "multiclass_data_2"

data = MNIST()

if not os.path.exists(folder):
    os.mkdir(folder)

X_train = data.train_data.reshape(-1, 28*28)
Y_train = np.argmax(data.train_labels, axis=1)
X_train, Y_train, permutation = helper.shuffleArraysInUnison(X_train, Y_train)

np.save(folder + "/" + "permutation.npy", permutation)

X_test = data.test_data.reshape(-1, 28*28)
Y_test = np.argmax(data.test_labels, axis=1)

train_size = len(X_train) / num_classifiers

models = []
training_sets = []
accs = []
for i in xrange(num_classifiers):
    start = train_size * i
    end = start + train_size
    training_sets.append([start, end])
    model = trainLMC(X_train[start:end], Y_train[start:end])
    acc = model.evaluate(X_test, Y_test)
    accs.append(acc)
    print("Model {}, Start {}, End {}, Test Accuracy {}".format(i, start, end, acc))
    np.save(folder + "/" + "weights_{}.npy".format(i), model.weights)
    np.save(folder + "/" + "bias_{}.npy".format(i), model.bias)
    models.append(model)

X_exp, Y_exp = helper.generate_data(num_points, X_test, Y_test, models)
print "Confirmation that selected points are all correctly classified ", [model.evaluate(X_exp, Y_exp)
                                                                          for model in models]

min_dists = helper.findNoiseBoundsMulti(models, X_exp, Y_exp)
print "Median Min Noise Bounds {}".format(np.median(min_dists))

np.save(folder + "/" + "test_accuracies.npy", np.array(accs))
np.save(folder + "/" + "dists.npy", np.array([min_dists]))
np.save(folder + "/" + "X_exp.npy", X_exp)
np.save(folder + "/" + "Y_exp.npy", Y_exp)
np.save(folder + "/" + "X_test.npy", X_test)
np.save(folder + "/" + "Y_test.npy", Y_test)
np.save(folder + "/" + "training_sets.npy", np.array(training_sets))
