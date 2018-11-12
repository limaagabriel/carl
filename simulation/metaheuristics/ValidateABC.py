from src.optimization.metaheuristics.single_objective.ABC import ABC
from src.optimization.problems.ObjectiveFunction import *
from src.optimization.problems.SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
from src.simulation.metaheuristics.Utils import create_dir
import numpy as np
import os


def main():
    search_space_initializer = OneQuarterDimWiseSSInitializer()
    #os.chdir('../../../..')
    file_path = "/home/clodomir/Desktop/Executions1/"
    num_exec = 30
    colony_size = 30
    num_iterations = 5000
    trials_limit = 100

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    dim = 30

    regular_functions = [SphereFunction, RosenbrockFunction, RastriginFunction, SchwefelFunction, SchafferFunction,
                         GriewankFunction, AckleyFunction, CigarFunction, HimmelblauFunction]

    cec_functions = [CECSphere, RotatedElliptic, BentCigar, RotatedDiscus, DifferentPowers, RotatedRosenbrock,
                     RotatedSchaffersF7, RotatedAckley, RotatedWeierstrass, RotatedGriewank, CECRastrigin,
                     RotatedRastrigin, NonContinuousRotatedRastrigin, CECSchwefel, RotatedSchwefel, RotatedKatsuura,
                     LunacekBiRastrigin, RotatedLunacekBiRastrigin, RotatedExpandedGriewankPlusRosenbrock,
                     RotatedExpandedScafferF6, CompositionFunction1, CompositionFunction2, CompositionFunction3,
                     CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7,
                     CompositionFunction8]

    for benchmark_func in cec_functions:
        func = benchmark_func(dim)
        run_experiments(num_iterations, colony_size, num_exec, trials_limit, func, search_space_initializer, file_path)


def run_experiments(n_iter, colony_size, num_runs, trials_limit, objective_function, search_space_initializer,
                    save_dir):
    alg_name = "ABC"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {:.2E}"
    if save_dir:
        create_dir(save_dir)
        f_handle_cost_iter = file(save_dir + "/ABC_" + objective_function.name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(save_dir + "/ABC_" + objective_function.name + "_cost_eval.txt", 'w+')

    for run in range(num_runs):
        opt1 = ABC(objective_function=objective_function, search_space_initializer=search_space_initializer,
                   n_iter=n_iter,
                   colony_size=colony_size, trials_limit=trials_limit)
        opt1.optimize()
        print console_out.format(alg_name, objective_function.name, run + 1, opt1.best_bee.cost)

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
