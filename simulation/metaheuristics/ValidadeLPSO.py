from src.optimization.metaheuristics.single_objective.LPSO import LPSO
from src.optimization.problems.ObjectiveFunction import *
from src.optimization.problems.SearchSpaceInitializer import OneQuarterDimWiseSSInitializer, UniformSSInitializer
import numpy as np


def main():
    search_space_initializer = OneQuarterDimWiseSSInitializer()
    file_path = "/home/clodomir/Desktop/Executions1/"
    num_exec = 30
    num_particles = 30
    num_iter = 5000
    lb_w = 0.72984
    up_w = 0.72984
    c1 = (0.72984 * 2.05)
    c2 = (0.72984 * 2.05)

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
        run_experiments(c1, c2, lb_w, num_iter, num_particles, num_exec, up_w, func, search_space_initializer,
                        file_path)


def run_experiments(c1, c2, lb_w, num_iter, num_particles, num_runs, up_w, objective_function, search_space_initializer,
                    save_dir):
    alg_name = "LPSO"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {:.2E}"
    if save_dir:
        f_handle_cost_iter = file(save_dir + "/LPSO_" + objective_function.name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(save_dir + "/LPSO_" + objective_function.name + "_cost_eval.txt", 'w+')

    for run in range(num_runs):
        opt1 = LPSO(objective_function, search_space_initializer, swarm_size=num_particles, n_iter=num_iter, lb_w=lb_w,
                    up_w=up_w, c1=c1, c2=c2, v_max=100)

        opt1.optimize()
        print console_out.format(alg_name, objective_function.name, run + 1, opt1.gbest_particle.cost)

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
