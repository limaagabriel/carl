import numpy as np


class BinaryProblems(object):
    def __init__(self, name, dim, minf, maxf, number_objectives):
        self.name = name
        self.minf = minf
        self.maxf = maxf
        self.number_objectives = number_objectives
        self.dim = dim

    def evaluate(self, x):
        pass


class OneMax(BinaryProblems):
    def __init__(self, dim):
        super(OneMax, self).__init__('ONE_MAX', dim, 0, 1, None)

    def evaluate(self, x):
        return sum(x)


class ZeroMax(BinaryProblems):
    def __init__(self, number_variables, number_objectives=2):
        super(ZeroMax, self).__init__('Zero Max', number_variables, 0, 1, number_objectives)

    def evaluate(self, val):
        return np.count_nonzero(np.asarray(val) == 0)


class KNAPSACK(BinaryProblems):
    A_10 = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    C_10 = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    S_10 = 269

    A_15 = [0.125126, 19.330424, 58.500931, 35.029145, 82.284005, 17.410810, 71.050142, 30.399487, 9.140294, 14.731285,
            98.852504, 11.908322, 0.891140, 53.166295, 60.176397]
    C_15 = [56.358531, 80.874050, 47.987304, 89.596240, 74.660482, 85.894345, 51.353496, 1.498459, 36.445204,
            16.589862, 44.569231, 0.466933, 37.788018, 57.118442, 60.716575]
    S_15 = 375

    A_20 = [92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58]
    C_20 = [44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75, 29, 75, 63]
    S_20 = 878

    A_23 = [981, 980, 979, 978, 977, 976, 487, 974, 970, 485, 485, 970, 970, 484, 484, 976, 974, 482, 962, 961, 959,
            958, 857]
    C_23 = [983, 982, 981, 980, 979, 978, 488, 976, 972, 486, 486, 972, 972, 485, 485, 969, 966, 483, 964, 963, 961,
            958, 959]
    S_23 = 10000

    A_50 = [220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125, 122, 120, 118, 115, 110, 105, 101, 100,
            100, 98, 96, 95, 90, 88, 82, 80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15, 10, 8,
            5, 3, 1]
    C_50 = [80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 50, 32, 22, 60, 30, 32, 40, 38, 35, 32, 25, 28, 3,
            22, 50, 30, 45, 30, 60, 50, 20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]
    S_50 = 1000

    A_100 = [297, 295, 293, 292, 291, 289, 284, 284, 283, 283, 281, 280, 279, 277, 276, 275, 273, 264, 260, 257, 250,
             236, 236, 235, 235, 233, 232, 232, 228, 218, 217, 214, 211, 208, 205, 204, 203, 201, 196, 194, 193, 193,
             192, 191, 190, 187, 187, 184, 184, 184, 181, 179, 176, 173, 172, 171, 160, 128, 123, 114, 113, 107, 105,
             101, 100, 100, 99, 98, 97, 94, 94, 93, 91, 80, 74, 73, 72, 63, 63, 62, 61, 60, 56, 53, 52, 50, 48, 46, 40,
             40, 35, 28, 22, 22, 18, 15, 12, 11, 6, 5]
    C_100 = [54, 95, 36, 18, 4, 71, 83, 16, 27, 84, 88, 45, 94, 64, 14, 80, 4, 23, 75, 36, 90, 20, 77, 32, 58, 6, 14,
             86, 84, 59, 71, 21, 30, 22, 96, 49, 81, 48, 37, 28, 6, 84, 19, 55, 88, 38, 51, 52, 79, 55, 70, 53, 64, 99,
             61, 86, 1, 64, 32, 60, 42, 45, 34, 22, 49, 37, 33, 1, 78, 43, 85, 24, 96, 32, 99, 57, 23, 8, 10, 74, 59,
             89, 95, 40, 46, 65, 6, 89, 84, 83, 6, 19, 45, 59, 26, 13, 8, 26, 5, 9]
    S_100 = 3820

    def __init__(self, dim):
        self.dim = dim
        dim_values = {10: [KNAPSACK.A_10, KNAPSACK.C_10, KNAPSACK.S_10],
                      15: [KNAPSACK.A_15, KNAPSACK.C_15, KNAPSACK.S_15],
                      20: [KNAPSACK.A_20, KNAPSACK.C_20, KNAPSACK.S_20],
                      23: [KNAPSACK.A_23, KNAPSACK.C_23, KNAPSACK.S_23],
                      50: [KNAPSACK.A_50, KNAPSACK.C_50, KNAPSACK.S_50],
                      100: [KNAPSACK.A_100, KNAPSACK.C_100, KNAPSACK.S_100]}
        if dim not in [10, 15, 20, 23, 50, 100]:
            raise Exception("Number of dimensions must be 10, 15, 23, 50 or 100")

        self.a, self.c, self.s = dim_values[self.dim]
        super(KNAPSACK, self).__init__('Knapsack', dim, 0, 1, None)

    def evaluate(self, positions):
        f = 0.0
        s_t = 0.0
        for i in range(self.dim):
            f += self.a[i] * positions[i]
            s_t += self.c[i] * positions[i]
        if s_t <= self.s:
            return f
        else:
            return 0.0
