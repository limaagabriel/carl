import math
import numpy as np


def SphereFunction(x):
    return (x ** 2)


def RosenbrockFunction(x):
    sum_ = 0.0

    for i in range(1, len(x) - 1):
        sum_ += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum_


def RastriginFunction(x):
    f_x = [(xi ** 2) - 10 * math.cos(2 * math.pi * xi) + 10 for xi in [x]]
    return sum(f_x)


def SchwefelFunction(x):
    sum_ = 0.0

    for i in range(0, len([x])):
        in_sum = 0.0
        for j in range(i):
            in_sum += x[j] ** 2
        sum_ += in_sum
    return sum_


def GeneralizedShwefelFunction(x):
    f_x = [xi * np.sin(np.sqrt(np.absolute(xi))) for xi in [x]]
    return -sum(f_x)


def GriewankFunction(x):
    fi = (1.0 / 4000) * np.sum(x ** 2)
    fii = 1.0

    for i in range(len([x])):
        fii *= np.cos(x / np.sqrt(i + 1))
    return fi + fii + 1


def AckleyFunction(x):

    exp_1 = -0.2 * np.sqrt((1.0 / len([x])) * (x ** 2))
    exp_2 = (1.0 / len([x])) * (np.cos(2 * math.pi * x))
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + math.e
