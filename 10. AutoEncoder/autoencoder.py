from sklearn.datasets import load_digits

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import sgd

import numpy as np


def main():
    dataset = load_digits()
    images = dataset['images']
    labels = dataset['target']
    del dataset

    tmp_images = np.zeros((len(images), 64))
    for image in range(len(images)):
        tmp_images[image] = images[image].flatten()
    images = tmp_images
    del tmp_images

    images /= 255.

    tmp_labels = np.zeros((len(labels), 10))
    for label in range(len(labels)):
        tmp_labels[label][labels[label]] = 1
    labels = tmp_labels
    del tmp_labels

    auto_encoder = auto_encode()
    auto_encoder.fit(images, images)
    weights = auto_encoder.layers[0].get_weights()
    train(images, labels, weights)


def auto_encode():
    model = Sequential()
    model.add(Dense(32, input_dim=64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    _sgd = sgd(lr=0.0228)
    model.compile(loss='mse', optimizer=_sgd, metrics=['accuracy'])
    return model


def train(images, labels, weights):
    model = Sequential()
    model.add(Dense(32, input_dim=64, init='uniform', weights=weights))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    _sgd = sgd(lr=0.0228)
    model.compile(loss='categorical_crossentropy', optimizer=_sgd, metrics=['accuracy'])
    model.fit(images, labels, validation_split=0.3, nb_epoch=1488)


if __name__ == '__main__':
    main()
