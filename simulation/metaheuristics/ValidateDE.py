from src.optimization.metaheuristics.single_objective.DE import DE
from src.optimization.problems.ObjectiveFunction import *
from src.optimization.problems.SearchSpaceInitializer import OneQuarterDimWiseSSInitializer, UniformSSInitializer
import pickle
import numpy as np


def main():
    search_space_initializer = OneQuarterDimWiseSSInitializer()
    file_path = "/home/clodomir/Desktop/Executions1/"

    dither_constant = 0.4
    population_size = 30
    num_iter = 100
    num_exec = 2
    n_cross = 1
    cr = 0.9

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    dim = 30

    regular_functions = [SphereFunction, RosenbrockFunction, RastriginFunction, SchwefelFunction, SchafferFunction,
                         GriewankFunction, AckleyFunction, CigarFunction, HimmelblauFunction]

    cec_functions = [RotatedRosenbrock,
                     RotatedSchaffersF7, RotatedAckley, RotatedWeierstrass, RotatedGriewank, CECRastrigin,
                     RotatedRastrigin, NonContinuousRotatedRastrigin, CECSchwefel, RotatedSchwefel, RotatedKatsuura,
                     LunacekBiRastrigin, RotatedLunacekBiRastrigin, RotatedExpandedGriewankPlusRosenbrock,
                     RotatedExpandedScafferF6, CompositionFunction1, CompositionFunction2, CompositionFunction3,
                     CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7,
                     CompositionFunction8]

    for benchmark_func in cec_functions:
        func = benchmark_func(dim)
        run_experiments(func, population_size, cr, n_cross, num_iter, dither_constant, num_exec,
                        file_path)


def run_experiments(objective_function, population_size, cr, n_cross, num_iter,  dither_constant, num_runs, save_dir):
    alg_name = "DE"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {}"
    if save_dir:
        f_handle_cost_iter = file(save_dir + "/DE_" + objective_function.name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(save_dir + "/DE_" + objective_function.name + "_cost_eval.txt", 'w+')

    for run in range(num_runs):
        opt1 = DE(objective_function, population_size, cr,  n_cross, num_iter, dither_constant)

        opt1.optimize()
        print console_out.format(alg_name, objective_function.name, run + 1, opt1.best_cost)

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
