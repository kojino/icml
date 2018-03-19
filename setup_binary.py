from setup_mnist import MNIST
import helper
import os
import numpy as np
from linear_models import trainLBC

classes = [4, 9]
num_classifiers = 100
num_points = 100
folder = "binary_data_2"

data = MNIST()
label_dict = dict(list(zip(classes, [-1, 1])))

if not os.path.exists(folder):
    os.mkdir(folder)

X_train, Y_train = helper.subsetData(data.train_data, data.train_labels, label_dict)
X_train, Y_train, permutation = helper.shuffleArraysInUnison(X_train, Y_train)

np.save(folder + "/" + "permutation.npy", permutation)

X_test, Y_test = helper.subsetData(data.test_data, data.test_labels, label_dict)

train_size = int(len(X_train) / num_classifiers)

models = []
training_sets = []
accs = []
for i in range(num_classifiers):
    start = train_size * i
    end = start + train_size
    print(start,end)
    training_sets.append([start, end])
    model = trainLBC(X_train[start:end], Y_train[start:end])
    acc = model.evaluate(X_test, Y_test)
    print(("Model {}, Start {}, End {}, Test Accuracy {}".format(i, start, end, acc)))
    np.save(folder + "/" + "weights_{}.npy".format(i), model.weights)
    np.save(folder + "/" + "bias_{}.npy".format(i), model.bias)
    models.append(model)
    accs.append(acc)

X_exp, Y_exp = helper.generate_data(num_points, X_test, Y_test, models)
print("Confirmation that selected points are all correctly classified ", [model.evaluate(X_exp, Y_exp)
                                                                          for model in models])
min_dists, max_dists = helper.findNoiseBoundsBinary(models, X_exp, Y_exp)
print("Median Min Max Noise Bounds {} {}".format(np.median(min_dists), np.median(max_dists)))

np.save(folder + "/" + "test_accuracies.npy", np.array(accs))
np.save(folder + "/" + "dists.npy", np.array([min_dists, max_dists]))
np.save(folder + "/" + "X_exp.npy", X_exp)
np.save(folder + "/" + "Y_exp.npy", Y_exp)
np.save(folder + "/" + "X_test.npy", X_test)
np.save(folder + "/" + "Y_test.npy", Y_test)
np.save(folder + "/" + "training_sets.npy", np.array(training_sets))
