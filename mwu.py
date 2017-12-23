import numpy as np
import time
import logging as log


def evaluateCosts(models, V, X, Y):
    return np.array([1 - model.evaluate(X + V, Y) for model in models])


def adversary(distribution, models, X, Y, alpha, noiseFunc):
    return np.array([noiseFunc(distribution, models, x, y, alpha) for x, y in zip(X,Y)])


def runMWU(models, T, X, Y, alpha, noiseFunc, exp_dir, epsilon=None):
    num_models = len(models)

    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(T))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    log.debug("\nRunning MWU for {} Iterations with Epsilon {}\n".format(T, epsilon))

    log.debug("Guaranteed to be within {} of the minimax value \n".format(delta))

    loss_history = []
    costs = []
    max_acc_history = []
    v = []
    w = []
    action_loss = []

    w.append(np.ones(num_models) / num_models)

    for t in xrange(T):
        log.debug("Iteration {}\n".format(t))

        if t % (T * .10) == 0 and t > 0:
            np.save(exp_dir + "/" + "weights_{}.npy".format(t), w)
            np.save(exp_dir + "/" + "noise_{}.npy".format(t), v)
            np.save(exp_dir + "/" + "loss_history_{}.npy".format(t), loss_history)
            np.save(exp_dir + "/" + "max_acc_history_{}.npy".format(t), max_acc_history)
            np.save(exp_dir + "/" + "action_loss_{}.npy".format(t), action_loss)

        start_time = time.time()

        v_t = adversary(w[t], models, X, Y, alpha, noiseFunc)
        v.append(v_t)

        cost_t = evaluateCosts(models, v_t, X, Y)
        costs.append(cost_t)

        avg_acc = np.mean((1 - np.array(costs)), axis=0)
        max_acc = max(avg_acc)
        max_acc_history.append(max_acc)

        loss = np.dot(w[t], cost_t)
        individual = [w[t][j] * cost_t[j] for j in xrange(num_models)]

        log.debug("Weights {} Sum of Weights {}".format(w[t], sum(w[t])))
        log.debug("Maximum (Average) Accuracy of Classifier {}".format(max_acc))
        log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y) for model in models])))
        log.debug("Cost (After Noise), {}".format(cost_t))
        log.debug("Loss {} Loss Per Action {}".format(loss, individual))

        loss_history.append(loss)
        action_loss.append(individual)

        new_w = np.copy(w[t])

        # penalize experts
        for i in xrange(num_models):
            new_w[i] *= (1.0 - epsilon) ** cost_t[i]

        # renormalize weights
        w_sum = new_w.sum()
        for i in xrange(num_models - 1):
            new_w[i] = new_w[i] / w_sum
        new_w[-1] = 1.0 - new_w[:-1].sum()

        w.append(new_w)

        log.debug("time spent {}\n".format(time.time() - start_time))
    log.debug("finished running MWU ")
    return w, v, loss_history, max_acc_history, action_loss
