from src.optimization.metaheuristics.single_objective.FA import FA
from src.optimization.problems.ObjectiveFunction import *
from src.optimization.problems.SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
from src.simulation.metaheuristics.Utils import create_dir
import numpy as np


def main():
    search_space_initializer = OneQuarterDimWiseSSInitializer()
    file_path = "/home/clodomir/Desktop/Executions1/"
    num_exec = 30
    population_size = 30
    num_iterations = 50000

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    dim = 100

    regular_functions = [SphereFunction, RosenbrockFunction, RastriginFunction, SchwefelFunction, SchafferFunction,
                         GriewankFunction, AckleyFunction, CigarFunction, HimmelblauFunction]

    # CECSphere, RotatedElliptic,
    cec_functions = [BentCigar, RotatedDiscus, DifferentPowers, RotatedRosenbrock,
                     RotatedSchaffersF7, RotatedAckley, RotatedWeierstrass, RotatedGriewank, CECRastrigin,
                     RotatedRastrigin, NonContinuousRotatedRastrigin, CECSchwefel, RotatedSchwefel, RotatedKatsuura,
                     LunacekBiRastrigin, RotatedLunacekBiRastrigin, RotatedExpandedGriewankPlusRosenbrock,
                     RotatedExpandedScafferF6, CompositionFunction1, CompositionFunction2, CompositionFunction3,
                     CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7,
                     CompositionFunction8]

    for benchmark_func in cec_functions:
        func = benchmark_func(dim)
        run_experiments(num_iterations, population_size, num_exec, func, search_space_initializer, file_path)


def run_experiments(n_iter, population_size, num_runs, objective_function, search_space_initializer, save_dir):
    alg_name = "FA"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {}"
    if save_dir:
        create_dir(save_dir)
        f_handle_cost_iter = file(save_dir + "/FA_" + objective_function.name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(save_dir + "/FA_" + objective_function.name + "_cost_eval.txt", 'w+')

    for run in range(num_runs):
        opt1 = FA(objective_function=objective_function, search_space_initializer=search_space_initializer, n_iter=n_iter,
                  population_size=population_size)

        opt1.optimize()
        print console_out.format(alg_name, objective_function.name, run + 1, opt1.best_firefly.cost)

        temp_optimum_cost_tracking_iter = np.asmatrix(opt1.optimum_cost_tracking_iter)
        temp_optimum_cost_tracking_eval = np.asmatrix(opt1.optimum_cost_tracking_eval)

        if save_dir:
            np.savetxt(f_handle_cost_iter, temp_optimum_cost_tracking_iter, fmt='%.4e')
            np.savetxt(f_handle_cost_eval, temp_optimum_cost_tracking_eval, fmt='%.4e')

    if save_dir:
        f_handle_cost_iter.close()
        f_handle_cost_eval.close()


if __name__ == '__main__':
    main()
