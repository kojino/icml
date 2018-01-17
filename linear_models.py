import numpy as np
from sklearn import svm
from helper import getMax
from noise_functions_multiclass import tryRegionOneVsAll



class LinearBinaryClassifier(object):
    """
    Class for Linear Binary Classifiers

    weights: np array of shape (dim, 1)
    bias: scalar
    """

    def __init__(self, weights, bias):
        self.dim = weights.shape[0]
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        """
        X: np array of shape (num_points, dim)

        returns: a vector of shape (num_points,) with predicted labels for each point
        """
        return np.sign(np.matmul(X, self.weights) + self.bias).T[0]

    def distance(self, X):
        """
        Computes the signed distance from a point to the decision boundary (hyperplane)

        returns: a vector of shape (num_points,) with the correspoding distances
        """
        return abs((np.matmul(X, self.weights) + self.bias) / np.linalg.norm(self.weights)).T[0]

    def evaluate(self, X, Y):
        """
        returns accuracy
        """
        return np.mean(np.equal(self.predict(X), Y))

    def gradient(self, X, Y):
        """
        returns gradient
        """
        if not hasattr(Y, "__len__"):  # make it robust to single items
            X = X.reshape(1, self.dim)
            Y = np.array([Y])

        return np.array([Y[i] * self.weights.reshape(-1, ) if self.predict(X[i]) == Y[i]
                         else np.zeros(self.dim) for i in xrange(len(X))])

    def rhinge_loss(self, X, Y):
        """
        returns average reverse hinge loss of classifier on X, Y

        defined as max{0, y(<w,x> + b)}
        """
        if not hasattr(Y, "__len__"):  # make it robust to single items
            X = X.reshape(1, self.dim)
            Y = np.array([Y])

        res = np.maximum(0, Y.reshape(-1, 1) * (np.matmul(X, self.weights) + self.bias))
        return np.mean(res.reshape(-1, ))


def trainLBC(X, Y):
    model = svm.SVC(kernel="linear")
    model.fit(X, Y)
    return LinearBinaryClassifier(model.coef_.T, model.intercept_)


class LinearOneVsAllClassifier(object):
    """
    Class for Lin Binary Classifiers

    weights: np array of shape (num_classes, dim)
    bias: scalar
    """

    def __init__(self, num_classes, weights, bias):
        self.dim = weights.shape[1]
        self.weights = weights
        self.bias = bias
        self.num_classes = num_classes

    def predict(self, X):
        """
        X: np array of shape (num_points, dim)

        returns: a vector of shape (num_points,) with predicted labels for each point
        """
        return np.argmax(np.matmul(X, self.weights.T) + self.bias, axis=1)

    def distance(self, X):
        """
        Computes the signed distance from a point to the decision boundary (hyperplane)

        returns: a vector of shape (num_points,) with the correspoding distances
        """
        n = X.shape[0]
        Y = self.predict(X)

        distances = []
        for i in xrange(n):
            label_options = range(10)
            del label_options[Y[i]]
            dists = []
            for j in label_options:
                v = tryRegionOneVsAll([self], [j], X[i])
                dists.append(np.linalg.norm(v))
            distances.append(min(dists))
        return distances

    def evaluate(self, X, Y):
        """
        returns accuracy
        """
        return np.mean(np.equal(self.predict(X), Y))

    def gradient(self, X, targets):
        """
        returns gradient
        """
        preds = np.matmul(X, self.weights.T) + self.bias
        n = X.shape[0]

        gradient = []

        for i in xrange(n):
            target = targets[i]
            others = range(10)
            del others[target]

            if np.argmax(preds[i]) == target:
                res = np.zeros(self.dim)
            else:
                max_ix = getMax(preds[i], target)[0]
                w_max = self.weights[max_ix]
                w_target = self.weights[target]
                res = w_max - w_target
            gradient.append(res)
        return np.array(gradient)

    def rhinge_loss(self, X, targets):
        preds = np.matmul(X, self.weights.T) + self.bias
        res = []
        for i in xrange(len(X)):
            target = targets[i]
            if np.argmax(preds) != target:
                max_ix, max_val = getMax(preds[i], target)
                loss = max_val - preds[i][target]
            else:
                loss = 0
            res.append(loss)
        return res

def trainLMC(X, Y):
    model = svm.LinearSVC(loss='hinge')
    model.fit(X, Y)
    return LinearOneVsAllClassifier(10, model.coef_, model.intercept_)
