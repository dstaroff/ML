from math import sqrt
from random import randint
from operator import itemgetter
from numpy import array, delete, alen


def distance(x, x0, features, metrics='euclid'):
    result = list()
    if metrics == 'euclid':
        for point in x:
            result.append((sqrt(sum([(point[features[i]] - x0[features[i]]) ** 2 for i in range(len(features))])),
                           point['kind']))
    elif metrics == 'manhattan':
        for point in x:
            result.append((sum([abs(point[features[i]] - x0[features[i]]) for i in range(len(features))]),
                           point['kind']))
    return result


def split_data(data_set):
    testing_data = list()
    count = int((len(data_set) * 30) / 100)
    data_copy = array(data_set)
    for i in range(count):
        index = randint(0, alen(data_copy) - 1)
        testing_data.append(data_copy[index])
        data_copy = delete(data_copy, index, 0)
    return data_copy, array(testing_data)


def knn(learning_data, test_point, k, features, weighted=False, metrics='euclid'):
    distances = distance(learning_data, test_point, features, metrics=metrics)
    k_nearest = array(sorted(distances)[:k])
    probabilities = dict()
    probability = 1
    for i in range(k):
        _, kind = k_nearest[i]
        if weighted:
            probability = (k - i) / k
        if kind in probabilities.keys():
            probabilities[kind] += probability
        else:
            probabilities[kind] = probability
    result = sorted(probabilities.items(), key=itemgetter(1))[0][0]
    return result


def cv(learning_data, testing_data, k, features, weighted=False, metrics='euclid'):
    error = 0
    for test_point in testing_data:
        experiment = knn(learning_data, test_point, k, features, weighted=weighted, metrics=metrics)
        if experiment != test_point['kind']:
            error += 1
    return error / alen(testing_data)


def grid_search(learning_data, testing_data, features, weighted=False, metrics='euclid'):
    grid = list()
    for find_k in range(1, int(sqrt(len(learning_data)))):
        error = cv(learning_data, testing_data, find_k, features, weighted=weighted, metrics=metrics)
        grid.append((error, find_k))
    return sorted(grid)[0][1]
