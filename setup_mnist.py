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
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD

# OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def extract_data(filemodel_dir, num_images):
    with gzip.open(filemodel_dir) as bytestream:
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


def train_network(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, file_name):
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=epochs)
    model.save(file_name)
    return model


def conv_net(extra_conv, num_layers, nodes_per_layer, restore=None):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if extra_conv:
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    for _ in xrange(num_layers):
        model.add(Dense(nodes_per_layer))
        model.add(Activation('relu'))
        model.add(Dense(nodes_per_layer))
        model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    if restore:
        model.load_weights(restore)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def multilayer(num_layers, nodes_per_layer, restore=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    for p in [nodes_per_layer] * num_layers:
        model.add(Dense(p))
        model.add(Activation('relu'))
        model.add(Dropout(.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    if restore:
        model.load_weights(restore)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    data = MNIST()
    model_dir = 'deep_networks'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    conv0 = conv_net(False, 2, 200)
    train_network(conv0, data.train_data, data.train_labels, data.validation_data, data.validation_labels, 32, 10,
                  model_dir + "/conv0")

    conv1 = conv_net(True, 2, 200)
    train_network(conv1, data.train_data, data.train_labels, data.validation_data, data.validation_labels, 32, 10,
                  model_dir + "/conv1")

    conv2 = conv_net(True, 4, 64)
    train_network(conv2, data.train_data, data.train_labels, data.validation_data, data.validation_labels, 32, 10,
                  model_dir + "/conv2")

    mlp0 = multilayer(4, 128)
    train_network(mlp0, data.train_data, data.train_labels, data.validation_data, data.validation_labels, 32, 10,
                  model_dir + "/mlp0")

    mlp1 = multilayer(2, 256)
    train_network(mlp1, data.train_data, data.train_labels, data.validation_data, data.validation_labels, 32, 10,
                  model_dir + "/mlp1")

