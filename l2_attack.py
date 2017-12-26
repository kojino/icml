## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Juan C. Perdomo 2017

import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results, default 1e-2
TARGETED = False         # should we target one specific class? or just be wrong?
CONFIDENCE = 0          # how strong the adversarial example should be
INITIAL_CONST = 1e2     # the initial constant c to pick as a first guess


class CarliniL2:
    def __init__(self, sess, models, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin=0.0, boxmax=255.0):
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
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.num_models = len(models)
        self.num_labels = num_labels

        shape = (batch_size, image_size, image_size, num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.weights = tf.Variable(np.zeros(self.num_models), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_weights = tf.placeholder(tf.float32, [self.num_models])

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        # prediction BEFORE-SOFTMAX of the model
        # np.expand_dims(self.newimg, axis=0)

        self.outputs = [model(self.newimg) for model in models]
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)), [1, 2, 3])
        
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
            print "targeted "
            # if targeted, optimize for making the other class most likely
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (others[i] - reals[i] + self.CONFIDENCE)))

        else:
            # if untargeted, optimize for making this class least likely.
            print "Untargeted "
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (reals[i] - others[i] + self.CONFIDENCE)))

        self.loss1list = loss1list  # TODO: remove

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * tf.add_n(self.loss1list))
        self.loss = self.loss1 + self.loss2
        self.reals = reals
        self.others = others

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.weights.assign(self.assign_weights))

        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targets, weights):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        # print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
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
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):

            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [0.0]*batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_weights: weights})

            # print "Outer Step ", outer_step, "Current C ", CONST, lower_bound, upper_bound

            print "GOT HERE "
            prev = 1e10 # used to be e6
            from keras import backend as K

            for iteration in xrange(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss, self.l2dist, self.outputs, self.newimg],
                                                        feed_dict={K.learning_phase(): 0})
                scores = np.array(scores).reshape(self.batch_size, self.num_models, self.num_labels)

                if iteration % 1000 == 0:
                    print "PREDICTION ", [np.argmax(score, axis=1) for score in scores]
                    print(iteration, self.sess.run((self.loss, self.loss1, self.loss2, self.loss1list, self.weights, self.reals, self.others),
                                                   feed_dict={K.learning_phase(): 0}))

                # check if we should abort search if we're getting nowhere. (check every 10%)
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS * .10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):

                    currLoss = compareLoss(sc, np.argmax(batchlab[e])) # expected loss of the learner

                    if currLoss > bestscore[e]:  # we've found a clear improvement for this value of c
                        bestl2[e] = l2
                        bestscore[e] = currLoss
                    if currLoss == bestscore[e] and l2 < bestl2[e]:
                        bestl2[e] = l2

                    if currLoss > o_bestscore[e]:
                        o_bestl2[e] = l2
                        o_bestscore[e] = currLoss
                        o_bestattack[e] = ii
                    if currLoss == o_bestscore[e] and l2 < o_bestl2[e]:
                        o_bestl2[e] = l2
                        o_bestattack[e] = ii

            # finished trying out the adam optimizer for a particular c, now need to decide on the next value
            # adjust the constant as needed
            for e in range(batch_size):
                if bestscore[e] == 1.0:
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 100

        # return the best solution found
        return o_bestattack
