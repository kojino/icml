import numpy as np
import time
import logging as log


def evaluateCosts(models, V, X, Y, targets, dl=False):
    """
    Returns the 0-1 loss of the models on input (X + V, Y) if targets is False
    Else returns the target accuracy of the models on (X + V, Targetes)
    dl is a bool to indicate whether the models are linear classifiers or deep learning models
    """
    if targets is not False:
        if dl:
            res = np.array([model.evaluate(X + V, targets)[1] for model in models])
        else:
            res = np.array([model.evaluate(X + V, targets) for model in models])
    else:
        if dl:
            res = np.array([1 - model.evaluate(X + V, Y)[1] for model in models])
        else:
            res = np.array([1 - model.evaluate(X + V, Y) for model in models])
    return res


def adversary(distribution, models, X, Y, alpha, noiseFunc, targets):
    """
    uses the noise function to compute adversarial perturbations that maximize the loss of the learner under
    the chosen distribution
    """
    if targets is not False:
        res = np.array([noiseFunc(distribution, models, x, y, alpha, target=target) for x, y, target
                        in zip(X, Y, targets)])
    else:
        res = np.array([noiseFunc(distribution, models, x, y, alpha) for x, y in zip(X, Y)])
    return res


def runMWU(models, T, X, Y, alpha, noiseFunc, exp_dir, epsilon=None, targeted=False, dl=False, sample=None, num_samples=10):
    if sample == 'once':
        # overwrite models by their sample
        models = np.random.choice(models,size=num_samples,replace=False)

    num_models = len(models)
    # compute epsilon as a function of the number of rounds, see MWU proof for more detail
    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(T))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    log.debug("\nRunning MWU for {} Iterations with Epsilon {}\n".format(T, epsilon))

    log.debug("Guaranteed to be within {} of the minimax value \n".format(delta))

    loss_history = []
    costs = []
    acc_history = []
    v = []
    w = []
    action_loss = []

    if sample == 'iter':
        w.append(np.ones(num_samples) / num_samples)
    else:
        w.append(np.ones(num_models) / num_models)

    for t in range(T):
        log.debug("Iteration {}\n".format(t))

        # samples num_sample models from `models`
        if sample == 'iter':
            # overwrite models by their sample
            sample_model_ids = np.random.choice(np.arange(num_samples),size=num_samples,replace=False)
            sampled_models = list(np.array(models)[sample_model_ids])

        if t % (T * .10) == 0 and t > 0:
            np.save(exp_dir + "/" + "weights_{}.npy".format(t), w)
            np.save(exp_dir + "/" + "noise_{}.npy".format(t), v)
            np.save(exp_dir + "/" + "loss_history_{}.npy".format(t), loss_history)
            np.save(exp_dir + "/" + "acc_history_{}.npy".format(t), acc_history)
            np.save(exp_dir + "/" + "action_loss_{}.npy".format(t), action_loss)

        start_time = time.time()

        # input sample_models instead of models
        if sample == 'iter':
            print(sampled_models)
            v_t = adversary(w[t], sampled_models, X, Y, alpha, noiseFunc, targeted)
            cost_t = evaluateCosts(sampled_models, v_t, X, Y, targeted, dl=dl)
        else:
            v_t = adversary(w[t], models, X, Y, alpha, noiseFunc, targeted)
            cost_t = evaluateCosts(models, v_t, X, Y, targeted, dl=dl)
        v.append(v_t)
        costs.append(cost_t)

        if targeted is not False:
            avg_loss = np.mean((np.array(costs)), axis=0)
            min_loss = min(avg_loss)
            acc_history.append(min_loss)
        else:
            avg_acc = np.mean((1 - np.array(costs)), axis=0)
            max_acc = max(avg_acc)
            acc_history.append(max_acc)

        loss = np.dot(w[t], cost_t)
        if sample == 'iter':
            individual = [w[t][j] * cost_t[j] for j in range(num_samples)]

        else:
            individual = [w[t][j] * cost_t[j] for j in range(num_models)]

        log.debug("Weights {} Sum of Weights {}".format(w[t], sum(w[t])))

        if targeted is not False:
            log.debug("Minimum (Average) Loss of Classifier {}".format(acc_history[-1]))
            if dl:
                log.debug("Cost (Before Noise) {}".format(np.array([model.evaluate(X, targeted)[1] for model in models])))
            else:
                log.debug("Cost (Before Noise) {}".format(np.array([model.evaluate(X, targeted) for model in models])))

        else:
            log.debug("Maximum (Average) Accuracy of Classifier {}".format(acc_history[-1]))
            if dl:
                log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y)[1] for model in models])))
            else:
                log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y) for model in models])))

        log.debug("Cost (After Noise), {}".format(cost_t))
        log.debug("Loss {} Loss Per Action {}".format(loss, individual))

        loss_history.append(loss)
        action_loss.append(individual)

        new_w = np.copy(w[t])

        # penalize experts
        # penalize according to the sampled models
        if sample == 'iter':
            avg_cost = np.mean(cost_t)
            for i,id in enumerate(sample_model_ids):
                new_w[id] *= (1.0 - epsilon) ** cost_t[i]
            for id in range(num_samples)-sample_model_ids:
                new_w[id] *= (1.0 - epsilon) ** avg_cost
        else:
            for i in range(num_models):
                new_w[i] *= (1.0 - epsilon) ** cost_t[i]

        # renormalize weights
        if sample == 'iter':
            w_sum = new_w.sum()
            for i in range(num_samples - 1):
                new_w[i] = new_w[i] / w_sum
            new_w[-1] = 1.0 - new_w[:-1].sum()
        else:
            w_sum = new_w.sum()
            for i in range(num_models - 1):
                new_w[i] = new_w[i] / w_sum
            new_w[-1] = 1.0 - new_w[:-1].sum()

        w.append(new_w)

        log.debug("time spent {}\n".format(time.time() - start_time))
    log.debug("finished running MWU ")
    return w, v, loss_history, acc_history, action_loss
