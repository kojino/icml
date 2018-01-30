## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import gzip
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf

OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def extract_data(file_name, num_images):
    with gzip.open(file_name) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255.0) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filemodel_dir, num_images):
    with gzip.open(filemodel_dir) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNIST:
    def __init__(self):
        train_data = extract_data("MNIST_data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("MNIST_data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("MNIST_data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

data = MNIST()

print data.train_labels[:3]
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))


# def fn(correct, predicted):
#     return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
#                                                    logits=predicted / 1.0)

model.compile(optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(data.train_data, data.train_labels, validation_data=(data.validation_data, data.validation_labels),
          batch_size=32, epochs=50, shuffle=True)