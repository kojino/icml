import numpy as np

from setup_mnist import MNIST, MNISTModel
from scipy.misc import imsave
import tensorflow as tf
from l2_attack import CarliniL2
from helper import generate_data, shuffleArraysInUnison
import time
import os

values = [range(10) for i in range(10)]
for i, val in enumerate(values):
    del val[i]

TARGET_DICT = {i: values[i] for i in range(10)}

if __name__ == "__main__":
    exp_dir = "test"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        os.mkdir(exp_dir+"/images")
        os.mkdir(exp_dir+"/images"+"/untargeted")
        os.mkdir(exp_dir+"/images"+"/targeted")
        os.mkdir(exp_dir+"/images"+"/normal")


    with tf.Session() as sess:

        data =  MNIST()
        inputs, labels = data.test_data, data.test_labels
        inputs, labels, permutation = shuffleArraysInUnison(inputs, labels)
        # np.save(exp_dir+"/"+"permutation.npy", permutation)



        models = [MNISTModel("models/mnist", sess),
                  MNISTModel("models/multilayer", sess, conv=False),
                  MNISTModel("models/mnist-distilled-100", sess),
                  MNISTModel("models/multilayer-distilled-100", sess, conv=False)]

        num_models = len(models)
        weights = [1.0 / num_models] * num_models

        print "Accuracy of Models"
        print [model.score(inputs, labels) for model in models]
        time.sleep(500m)


        untargeted = CarliniL2(sess, models, targeted=False, batch_size=1,
                            max_iterations=4000, binary_search_steps=9, confidence=0)

        num_untargeted = 50
        unt_inputs, unt_labels = generate_data(num_untargeted, inputs, labels, models)

        print "Weights ", weights
        unt_s = time.time()
        untargeted_adv = untargeted.attack(unt_inputs, unt_labels, weights)
        print "Time in Untargeted Attacks ", time.time() - unt_s
        untargeted_results = []

        for i in xrange(len(untargeted_adv)):
            true_label = np.argmax(unt_labels[i])

            distortion = np.sum((untargeted_adv[i] - unt_inputs[i]) ** 2) ** .5

            predicted_labels = [np.argmax(model.model.predict(untargeted_adv[i:i+1])) for model in models]
            success_vector = (np.array(predicted_labels) != true_label).astype(np.float32)

            res = [i, true_label, distortion] + predicted_labels + list(success_vector)
            untargeted_results.append(res)

            imsave(exp_dir+"/images/normal/"+str(i)+"_"+str(true_label)+"_normal.jpg", unt_inputs[i].reshape(28, 28))
            imsave(exp_dir+"/images/untargeted/"+str(i)+"_"+str(true_label)+"_untargeted.jpg", untargeted_adv[i].reshape(28, 28))

        np.save(exp_dir+"/untargeted_results.npy", np.array(untargeted_results))

        print "UnTargeted Results"
        for row in untargeted_results:
            print row


        targeted = CarliniL2(sess, models, targeted=True, batch_size=1,
                             max_iterations=4000, binary_search_steps=9, confidence=0)

        num_targeted = 8
        t_inputs, t_labels = generate_data(num_targeted, inputs, labels, models, target_dict=TARGET_DICT)

        t_s = time.time()
        targeted_adv = targeted.attack(t_inputs, t_labels, weights)
        print "Time in Targeted Attacks ", time.time() - t_s
        targeted_results = []

        for i in xrange(len(targeted_adv)):
            j = i // 9
            true_label = np.argmax(models[0].model.predict(t_inputs[i:i+1]))
            target_label = np.argmax(t_labels[i])

            distortion = np.sum((targeted_adv[i] - t_inputs[i]) ** 2) ** .5

            predicted_labels = [np.argmax(model.model.predict(targeted_adv[i:i+1])) for model in models]
            success_vector = (np.array(predicted_labels) == target_label).astype(np.float32)

            res = [i, j, true_label, target_label, distortion] + predicted_labels + list(success_vector)
            targeted_results.append(res)

            imsave(exp_dir+"/images/targeted/"+str(i)+"_"+str(j)+"_"+str(true_label)+"_"+str(target_label)+"_targeted.jpg",
                   targeted_adv[i].reshape(28, 28))

        print "Targeted Results"
        for row in targeted_results:
            print row

        np.save(exp_dir+"/targeted_results.npy", np.array(targeted_results))

        print "Success :)"

