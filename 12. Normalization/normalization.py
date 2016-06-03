from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import sgd

from pickle import load


def main():
    images = load(open('images', 'rb'))
    labels = load(open('labels', 'rb'))

    train(images, labels)


def train(images, labels):
    model = Sequential()
    model.add(Dense(64, input_dim=64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    _sgd = sgd(lr=0.0228, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=_sgd, metrics=['accuracy'])
    model.fit(images, labels, validation_split=0.3, nb_epoch=1488, batch_size=750)


if __name__ == '__main__':
    main()
