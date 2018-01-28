## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Juan C. Perdomo 2017

import tensorflow as tf
from tensorflow.contrib.opt import VariableClippingOptimizer
import numpy as np
import logging as log

MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results, default 1e-2
TARGETED = False         # should we target one specific class? or just be wrong?
CONFIDENCE = 0          # how strong the adversarial example should be

class CarliniL2:
    def __init__(self, sess, models, alpha, batch_size=1, confidence=CONFIDENCE, targeted=TARGETED,
                 learning_rate=LEARNING_RATE, max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 min_val=0.0, max_val=255.0):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = 224, 3, 1000  # imagenet parameters
        self.sess = sess
        self.alpha = alpha
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.num_models = len(models)
        self.num_labels = num_labels
        self.min_val = min_val
        self.max_val = max_val

        shape = (batch_size, image_size, image_size, num_channels)
        
        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32)
        self.weights = tf.Variable(np.zeros(self.num_models), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_weights = tf.placeholder(tf.float32, [self.num_models])

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        # self.boxmul = (boxmax - boxmin) / 2.
        # self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.clip_by_value(self.modifier + self.timg, self.min_val, self.max_val)

        # prediction BEFORE-SOFTMAX of the model
        # np.expand_dims(self.newimg, axis=0)

        self.outputs = [model(self.newimg) for model in models]
        
        # distance to the input data
        # self.l2dist = tf.reduce_sum(tf.square(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)),
        #                             [1, 2, 3])
        self.norm = tf.norm(self.modifier)
        
        # compute the probability of the label class versus the maximum other
        reals = []
        others = []
        for i in xrange(self.num_models):
            real = tf.reduce_sum(self.tlab * self.outputs[i], 1)
            other = tf.reduce_max((1 - self.tlab) * self.outputs[i], 1)
            reals.append(real)
            others.append(other)
        self.reals, self.others = reals, others

        loss1list = []

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (others[i] - reals[i] + self.CONFIDENCE)))
        else:
            # if untargeted, optimize for making this class least likely.
            # print "Untargeted "
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (reals[i] - others[i] + self.CONFIDENCE)))

        self.loss1list = loss1list

        # sum up the losses
        # self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.add_n(self.loss1list)
        self.loss = self.loss1
        self.reals = reals
        self.others = others

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        # adam = tf.train.AdamOptimizer(self.LEARNING_RATE)
        sgd = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        optimizer = VariableClippingOptimizer(sgd, {self.modifier: [1, 2, 3]}, self.alpha)
        self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        # print "ALPHHHAAAAA", self.alpha
        # print  "SHAPE ", self.modifier.shape
        # self.clip = tf.assign(self.modifier, tf.clip_by_norm(self.modifier, self.alpha))

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.weights.assign(self.assign_weights))

        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)

    def attack(self, imgs, targets, weights):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        # print('go up to',len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            # print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size], weights))
        return np.array(r)

    def attack_batch(self, imgs, labs, weights):
        """
        Run the attack on a batch of images and labels.
        """
        def compareLoss(x, y):
            """
            x is an np array of shape num_models x num_classes
            y is the true label or target label of the class

            returns a number in [0,1] indicating the expected loss of the learner
            """
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                for v in x:  # update the target scores for each individual prediction
                    if self.TARGETED:
                        v[y] -= self.CONFIDENCE
                    else:
                        v[y] += self.CONFIDENCE
                x = np.argmax(x, 1)  # these are the predictions of each hypothesis

            if self.TARGETED:
                return np.dot(x == y, weights)
            else:
                return np.dot(x != y, weights)

        batch_size = self.batch_size

        # convert to tanh-space
        # imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        #
        # # set the lower and upper bounds accordingly
        # lower_bound = np.zeros(batch_size)
        # CONST = np.ones(batch_size) * self.initial_const
        # upper_bound = np.ones(batch_size) * 1e10

        # # the best l2, score, and image attack
        # o_bestl2 = [1e10]*batch_size
        # o_bestscore = [-1]*batch_size
        bestattack = [np.zeros(imgs[0].shape)] * batch_size
        
        # for outer_step in range(self.BINARY_SEARCH_STEPS):

        # completely reset adam's internal state.
        self.sess.run(self.init)
        batch = imgs[:batch_size]
        batchlab = labs[:batch_size]

        # bestl2 = [1e10]*batch_size
        bestscore = [0.0] * batch_size

        # set the variables so that we don't have to send them over again
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_tlab: batchlab,
                                   self.assign_weights: weights})

        from keras import backend as K

        for iteration in xrange(self.MAX_ITERATIONS):

            # perform the attack
            self.sess.run([self.train], feed_dict={K.learning_phase(): 0})
            # norm = self.sess.run([self.norm], feed_dict={K.learning_phase(): 0})

            norm, loss, scores, nimg = self.sess.run([self.norm, self.loss1list, self.outputs, self.newimg],
                                                    feed_dict={K.learning_phase(): 0})
            if iteration % 10 == 0:
                print "Iteration ", iteration
                print "NORM ", norm
                print "LOSS ", loss
                print

            scores = np.array(scores).reshape(self.batch_size, self.num_models, self.num_labels)

            # if iteration % step_print == 0:
            #     print "PREDICTION ", [np.argmax(score, axis=1) for score in scores]

            # check if we should abort search if we're getting nowhere. (check every 10%)
            # if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS * .10) == 0:
            #     if l > prev*.9999:
            #         break # TODO
            #     prev = l

            for e, (sc, ii) in enumerate(zip(scores, nimg)):

                currLoss = compareLoss(sc, np.argmax(batchlab[e])) # expected loss of the learner

                if currLoss > bestscore[e]:  # we've found a clear improvement for this value of c
                    bestscore[e] = currLoss
                    bestattack[e] = ii

        # return the best solution found
        return bestattack
