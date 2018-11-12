import numpy as np

class MultiobjectiveFunction(object):
    def __init__(self, name, number_variables, minf, maxf,number_objectives):
        self.function_name = name
        self.minf = minf
        self.maxf = maxf
        self.number_objectives = number_objectives
        self.dim = number_variables

    def evaluate(self, x):
        pass


class ZDT1(MultiobjectiveFunction):
    def __init__(self, number_variables=30):
        super(ZDT1, self).__init__('ZDT1', number_variables, np.zeros(number_variables), np.ones(number_variables), 2)

    def evaluate(self, x):
        f = np.zeros(self.number_objectives)
        f[0] = x[0]
        g = self.__evalG__(x)
        h = self.__evalH__(f[0], g)
        f[1] = h * g
        return f


    def __evalG__(self,x):
        g = 0.0
        for i in range(self.dim):
            g += x[i]

        constant = (9.0 / (self.dim - 1))
        g = constant * g
        g = g + 1.0
        return g

    def __evalH__(self,f,g):
        h = 0.0
        h = 1.0 - np.sqrt(f / g);
        return h;
