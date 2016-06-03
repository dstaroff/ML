from math import sin, cosh, cos, exp, sqrt
from pickle import load
from random import randint
from sys import argv

from matplotlib.pylab import frange
from numpy import array, min, max, dot, delete, insert
from numpy.linalg import inv

from plot import plot


def find_betas(x, y):
    """
    :param x: matrix of scalar points: X
    :param y: matrix of values: Y(x) = Beta * X
    :return: matrix of betas: Beta = (X` * X) ** (-1) * X` * Y
    """
    return dot(dot(inv(dot(x.transpose(), x)), x.transpose()), y)


def linear(data, kwargs='for_cv'):
    """
    Linear interpolation of points set
    :param data: List of X, Y
    :param kwargs: Impacts on returning value.
        If 'for_cv, then returns linear function for given data.
        If 'for_plot', then returns standard triple for PLOT function.
        Default: 'for_cv'.
    :return: Initial points set, X, Y(X)
    """
    x, y = data

    x_min = min(x)
    x_max = max(x)
    points = [x, y]

    x = array([[1, x[i][0]] for i in range(len(x))])

    beta = find_betas(x, y)

    if kwargs == 'for_cv':
        def func(arg):
            return beta[0][0] + beta[1][0] * arg

        return func
    elif kwargs == 'for_plot':
        abscissa = frange(x_min, x_max, 0.01)
        ordinate = [beta[0][0] + beta[1][0] * arg for arg in abscissa]

        return points, abscissa, ordinate


def polynomial(data, k, kwargs='for_cv'):
    """
    Polynomial interpolation of points set
    :param data: List of X, Y
    :param k: Degree of polynomial
    :param kwargs: Impacts on returning value.
        If 'for_cv, then returns polynomial function for given data.
        If 'for_plot', then returns standard triple for PLOT function.
        Default: 'for_cv'.
    :return: Initial points set, X, Y(X)
    """
    x, y = data

    x_min = min(x)
    x_max = max(x)
    points = [x, y]

    # Making a Vandermonde's matrix
    x_temp = [list() for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(k + 1):
            x_temp[i].append(x[i][0] ** j)
    x = array(x_temp)
    del x_temp

    beta = find_betas(x, y)

    if kwargs == 'for_cv':
        def func(arg):
            value = 0
            for j in range(k + 1):
                value += beta[j][0] * (arg ** j)
            return value

        return func
    elif kwargs == 'for_plot':
        abscissa = frange(x_min, x_max, 0.01)
        ordinate = list()
        for arg in abscissa:
            value = 0
            for j in range(k + 1):
                value += beta[j][0] * (arg ** j)
            ordinate.append(value)

        return points, abscissa, ordinate


def non_linear(data, functions, kwargs='for_cv'):
    """
    Non-linear interpolation of points set
    :param data: List of X, Y
    :param functions: List of functions
    :param kwargs: Impacts on returning value.
        If 'for_cv, then returns polynomial function for given data.
        If 'for_plot', then returns standard triple for PLOT function
        Default: 'for_cv'
    :return: Initial points set, X, Y(X)
    """
    x, y = data

    x_min = min(x)
    x_max = max(x)
    points = [x, y]

    # Making a Vandermonde's matrix of functions
    functions.insert(0, lambda arg: arg)
    functions.insert(0, lambda arg: 1)
    x_temp = [list() for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(functions)):
            x_temp[i].append(functions[j](x[i][0]))
    x = array(x_temp)
    del x_temp

    beta = find_betas(x, y)

    if kwargs == 'for_cv':
        def func(arg):
            value = 0
            for j in range(len(functions)):
                value += beta[j][0] * functions[j](arg)
            return value

        return func
    elif kwargs == 'for_plot':
        abscissa = frange(x_min, x_max, 0.01)
        ordinate = list()
        for arg in abscissa:
            value = 0
            for j in range(len(functions)):
                value += beta[j][0] * functions[j](arg)
            ordinate.append(value)

        return points, abscissa, ordinate


def leave_one_out(data, t, k=5, functions=[]):
    mistake = 0
    x, y = data
    count = len(x)
    for _ in range(count // k):
        index = randint(0, len(x) - 1)
        checkpoint = [x[index], y[index]]
        x = delete(x, index, 0)
        y = delete(y, index, 0)
        p = None
        if t == '--linear':
            p = linear([x, y])
        elif t == '--polynomial':
            p = polynomial([x, y], k)
        elif t == '--nonlinear':
            p = non_linear([x, y], functions.copy())
        insert(x, index, checkpoint[0])
        insert(y, index, checkpoint[1])
        mistake += abs(checkpoint[1][0] - p(checkpoint[0][0]))
    return mistake / count


def monte_carlo(data, t, n, k=5, functions=[]):
    mistake = 0
    x, y = data
    count = len(x)
    checkpoints = list()
    for i in range((count - 10) // n):
        for j in range(n):
            index = randint(0, len(x) - 1)
            checkpoint = [x[index], y[index]]
            checkpoints.append(checkpoint)
            x = delete(x, index, 0)
            y = delete(y, index, 0)
        p = None
        if t == '--linear':
            p = linear([x, y])
        elif t == '--polynomial':
            p = polynomial([x, y], k)
        elif t == '--nonlinear':
            p = non_linear([x, y], functions.copy())

        local_mistake = 0
        for j in range(n):
            local_mistake += abs(checkpoints[j][1] - p(checkpoints[j][0]))
        local_mistake /= n
        mistake += local_mistake
    return mistake / count


def cv(data, regression_type, k=5, functions=[]):
    print('{0}:\n\tleave-one-out: {1}\n\tmonte-carlo: {2}'.format(regression_type,
                                                                  leave_one_out(data,
                                                                                regression_type,
                                                                                k=k,
                                                                                functions=functions),
                                                                  monte_carlo(data,
                                                                              regression_type,
                                                                              5,
                                                                              k=k,
                                                                              functions=functions)[0]))


def main(k=5, dataset_number=2, types='--all', functions=[]):
    file = 'task2_dataset_{0}.txt'.format(dataset_number)
    data = load(open(file, 'rb'), encoding='bytes')
    if types == '--all':
        cv(data, '--linear')
        cv(data, '--polynomial', k=k)
        cv(data, '--nonlinear', functions=functions)
        plot([linear(data, kwargs='for_plot'),
              polynomial(data, k, kwargs='for_plot'),
              non_linear(data, functions, kwargs='for_plot')],
             ['Linear', 'Polynomial', 'Non-Linear'])
    elif types == '--linear':
        cv(data, '--linear')
        plot([linear(data, kwargs='for_plot')], title=['Linear'])
    elif types == '--polynomial':
        cv(data, '--polynomial', k=k)
        plot([polynomial(data, k, kwargs='for_plot')], title=['Polynomial'])
    elif types == '--nonlinear':
        cv(data, '--linear', functions=functions)
        plot([non_linear(data, functions, kwargs='for_plot')], title=['Non-Linear'])


if __name__ == '__main__':
    tps = '--all'
    data_set_number = 2
    k = 5
    funcs = [lambda arg: sin(arg),
             lambda arg: cosh(arg),
             lambda arg: cos(arg),
             lambda arg: exp(arg),
             lambda arg: sqrt(abs(arg))]
    if len(argv) > 1:
        tps = argv[1]
        if len(argv) > 2:
            data_set_number = argv[2]
            if len(argv) > 3:
                k = int(argv[3])
    main(k=k, dataset_number=data_set_number, types=tps, functions=funcs)
