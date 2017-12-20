import numpy as np
from sklearn import svm


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


