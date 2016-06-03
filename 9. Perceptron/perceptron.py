from sklearn.datasets import load_digits

from keras.models import Sequential
from keras.layers.core import Dense, Activation

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

    train(images, labels)


def train(images, labels):
    model = Sequential()
    model.add(Dense(64, input_dim=64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(images, labels, validation_split=0.3, show_accuracy=True)


if __name__ == '__main__':
    main()
