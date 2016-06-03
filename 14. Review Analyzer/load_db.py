from keras.datasets.imdb import load_data
from pickle import dump


if __name__ == '__main__':
    data = load_data(nb_words=20000, test_split=0.3)
    dump(data, open('imdb', 'wb'))
