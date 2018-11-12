from deap.benchmarks import sphere, rosenbrock, rastrigin, schwefel, schaffer, griewank, ackley, cigar, himmelblau, h1
from deap.benchmarks.tools import translate, rotate
from cec2013single.cec2013 import Benchmark

# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy


class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf, rot=None, trans=None):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf
        self.rot = rot
        self.trans = trans

    def evaluate(self, x):
        pass


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim, rot=None, trans=None):
        super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0, rot, trans)
        self.func = sphere
        if self.rot:
            rotation = rotate(self.rot)
            self.func = rotation(self.func)

        if self.trans:
            translation = translate(self.trans)
            self.func = translation(self.func)

    def evaluate(self, x):
        return self.func(x)[0]


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    def evaluate(self, x):
        return rosenbrock(x)[0]


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def evaluate(self, x):
        return rastrigin(x)[0]


class SchwefelFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SchwefelFunction, self).__init__('Schwefel', dim, -30.0, 30.0)

    def evaluate(self, x):
        return schwefel(x)[0]


class SchafferFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SchafferFunction, self).__init__('Schaffer', dim, -100.0, 100.0)

    def evaluate(self, x):
        return schaffer(x)[0]


class GriewankFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(GriewankFunction, self).__init__('Griewank', dim, -600.0, 600.0)

    def evaluate(self, x):
        return griewank(x)[0]


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(AckleyFunction, self).__init__('Ackley', dim, -32.0, 32.0)

    def evaluate(self, x):
        return ackley(x)[0]


class CigarFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(CigarFunction, self).__init__('Cigar', dim, -100.0, 100.0)

    def evaluate(self, x):
        return cigar(x)[0]


class HimmelblauFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(HimmelblauFunction, self).__init__('Himmelblau', dim,  -6, 6)

    def evaluate(self, x):
        return himmelblau(x)[0]


class CECSphere(ObjectiveFunction):
    def __init__(self, dim):
        super(CECSphere, self).__init__('CEC Sphere', dim, -100, 100)
        self.func = Benchmark().get_function(1)

    def evaluate(self, x):
        return self.func(x)


class RotatedElliptic(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedElliptic, self).__init__('Rotated Elliptic', dim, -100, 100)
        self.func = Benchmark().get_function(2)

    def evaluate(self, x):
        return self.func(x)


class BentCigar(ObjectiveFunction):
    def __init__(self, dim):
        super(BentCigar, self).__init__('Bent Cigar', dim, -100, 100)
        self.func = Benchmark().get_function(3)

    def evaluate(self, x):
        return self.func(x)


class RotatedDiscus(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedDiscus, self).__init__('Rotated Discus', dim, -100, 100)
        self.func = Benchmark().get_function(4)

    def evaluate(self, x):
        return self.func(x)


class DifferentPowers(ObjectiveFunction):
    def __init__(self, dim):
        super(DifferentPowers, self).__init__('Different Powers', dim, -100, 100)
        self.func = Benchmark().get_function(5)

    def evaluate(self, x):
        return self.func(x)


class RotatedRosenbrock(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedRosenbrock, self).__init__('Rotated Rosenbrock', dim, -100, 100)
        self.func = Benchmark().get_function(6)

    def evaluate(self, x):
        return self.func(x)


class RotatedSchaffersF7(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedSchaffersF7, self).__init__('Rotated Schaffers F7', dim, -100, 100)
        self.func = Benchmark().get_function(7)

    def evaluate(self, x):
        return self.func(x)


class RotatedAckley(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedAckley, self).__init__('Rotated Ackley', dim, -100, 100)
        self.func = Benchmark().get_function(8)

    def evaluate(self, x):
        return self.func(x)


class RotatedWeierstrass(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedWeierstrass, self).__init__('Rotated Weierstrass', dim, -100, 100)
        self.func = Benchmark().get_function(9)

    def evaluate(self, x):
        return self.func(x)


class RotatedGriewank(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedGriewank, self).__init__('Rotated Griewank', dim, -100, 100)
        self.func = Benchmark().get_function(10)

    def evaluate(self, x):
        return self.func(x)


class CECRastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(CECRastrigin, self).__init__('CEC Rastrigin', dim, -100, 100)
        self.func = Benchmark().get_function(11)

    def evaluate(self, x):
        return self.func(x)


class RotatedRastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedRastrigin, self).__init__('Rotated Rastrigin', dim, -100, 100)
        self.func = Benchmark().get_function(12)

    def evaluate(self, x):
        return self.func(x)


class NonContinuousRotatedRastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(NonContinuousRotatedRastrigin, self).__init__('Non-continuous Rotated Rastrigin', dim, -100, 100)
        self.func = Benchmark().get_function(13)

    def evaluate(self, x):
        return self.func(x)


class CECSchwefel(ObjectiveFunction):
    def __init__(self, dim):
        super(CECSchwefel, self).__init__('CEC Schwefel', dim, -100, 100)
        self.func = Benchmark().get_function(14)

    def evaluate(self, x):
        return self.func(x)


class RotatedSchwefel(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedSchwefel, self).__init__('Rotated Schwefel', dim, -100, 100)
        self.func = Benchmark().get_function(15)

    def evaluate(self, x):
        return self.func(x)


class RotatedKatsuura(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedKatsuura, self).__init__('Rotated Katsuura', dim, -100, 100)
        self.func = Benchmark().get_function(16)

    def evaluate(self, x):
        return self.func(x)


class LunacekBiRastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(LunacekBiRastrigin, self).__init__('Lunacek bi-Rastrigin', dim, -100, 100)
        self.func = Benchmark().get_function(17)

    def evaluate(self, x):
        return self.func(x)


class RotatedLunacekBiRastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedLunacekBiRastrigin, self).__init__('Rotated Lunacek Bi-Rastrigin', dim, -100, 100)
        self.func = Benchmark().get_function(18)

    def evaluate(self, x):
        return self.func(x)


class RotatedExpandedGriewankPlusRosenbrock(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedExpandedGriewankPlusRosenbrock, self).__init__('Rotated Expanded Griewank Plus Rosenbrock', dim, -100, 100)
        self.func = Benchmark().get_function(19)

    def evaluate(self, x):
        return self.func(x)


class RotatedExpandedScafferF6(ObjectiveFunction):
    def __init__(self, dim):
        super(RotatedExpandedScafferF6, self).__init__('Rotated Expanded Scaffer F6', dim, -100, 100)
        self.func = Benchmark().get_function(20)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction1(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction1, self).__init__('Composition Function 1', dim, -100, 100)
        self.func = Benchmark().get_function(21)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction2(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction2, self).__init__('Composition Function 2', dim, -100, 100)
        self.func = Benchmark().get_function(22)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction3(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction3, self).__init__('Composition Function 3', dim, -100, 100)
        self.func = Benchmark().get_function(23)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction4(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction4, self).__init__('Composition Function 4', dim, -100, 100)
        self.func = Benchmark().get_function(24)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction5(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction5, self).__init__('Composition Function 5', dim, -100, 100)
        self.func = Benchmark().get_function(25)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction6(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction6, self).__init__('Composition Function 6', dim, -100, 100)
        self.func = Benchmark().get_function(26)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction7(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction7, self).__init__('Composition Function 7', dim, -100, 100)
        self.func = Benchmark().get_function(27)

    def evaluate(self, x):
        return self.func(x)


class CompositionFunction8(ObjectiveFunction):
    def __init__(self, dim):
        super(CompositionFunction8, self).__init__('Composition Function 8', dim, -100, 100)
        self.func = Benchmark().get_function(28)

    def evaluate(self, x):
        return self.func(x)