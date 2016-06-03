from __future__ import print_function

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences

from pickle import load

import numpy as np


def train(x_train, x_test, y_train, y_test, max_features, batch_size, max_len):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(1))
    model.add(Activation(activation='sigmoid'))

    print('\tModel is compiling')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('\tModel is fitting')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score, acc


def main():
    np.random.seed(1488)
    max_features = 20000
    max_len = 80
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = load(open('imdb', 'rb'))
    print('Data is loaded')
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    print('Data is pad sequenced\nTraining:')
    score, acc = train(x_train, x_test, y_train, y_test, max_features, batch_size, max_len)
    print('Score: {0}\nAccuracy: {1}'.format(score, acc))


if __name__ == '__main__':
    main()
