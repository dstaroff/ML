from math import sqrt
from random import random, randint

from numpy import array


class GradientDescent:
    def __init__(self, x0, a, b):
        self.x0 = x0
        self.a = a
        self.b = b
        self.n = len(x0)

    def distance(self, a, b):
        return sqrt(sum([(a[i] - b[i]) ** 2 for i in range(self.n)]))

    def dichotomy(self, function):
        a = array([0] * self.n)
        b = array([0.1] * self.n)
        delta = 0.01
        count = 2 ** 5
        for j in range(count):
            mid = (a + b) / 2
            x1 = mid - delta
            x2 = mid + delta
            gradient = function(self.x0, self.n, gradient=True)
            func1 = function(self.x0 - x1 * gradient, self.n)
            func2 = function(self.x0 - x2 * gradient, self.n)
            if func1 > func2:
                a = x1
            else:
                b = x2
        return (a + b) / 2

    def monte_carlo(self, function, eps):
        N = 10 ** 4
        mini = float('inf')
        x = array([0] * self.n)
        x1 = array([0] * self.n)
        lambd = self.dichotomy(function)
        for i in range(N):
            x0 = array([random() * randint(self.a, self.b)] * self.n)
            x1 = x0 - lambd * function(x0, self.n, gradient=True)
            while self.distance(x0, x1) > eps:
                x0 = x1
                x1 = x0 - lambd * function(x0, self.n, gradient=True)
            if function(x1, self.n) < mini:
                mini = function(x1, self.n)
                x = x1
        return x


def f(x, n, gradient=False):
    if not gradient:
        return x ** 3 - 4 * x ** 2 + 2 * x
    else:
        return 3 * x ** 2 - 8 * x + 2


def f1(x, n, gradient=False):
    if not gradient:
        return sum([(1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2 for i in range(n - 1)])
    else:
        grad = array([0] * n)
        grad[0] = (-2) * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0]
        for i in range(1, n):
            if i != n - 1:
                grad[i] = 200 * (x[i] - x[i - 1] ** 2) - 2 * (1 - x[i]) - 400 * (x[i + 1] - x[i] ** 2) * x[i]
            else:
                grad[i] = 200 * (x[i] - x[i - 1] ** 2)
        return grad


def main(function, x0, a, b, lambd, epsilon, t):
    gr = GradientDescent(x0=x0, a=a, b=b)
    if t != 'monte-carlo':
        const = 0.9
        x1 = x0 - lambd * function(x0, len(x0), gradient=True)
        while gr.distance(x0, x1) > epsilon:
            if t == 'constant-descent':
                lambd *= const
            elif t == 'optimal-descent':
                lambd = gr.dichotomy(function)
            for i in range(len(x0)):
                x0 = x1
                x1 = x0 - lambd * function(x0, len(x0), gradient=True)
    else:
        x1 = gr.monte_carlo(function, epsilon)
    return x1
