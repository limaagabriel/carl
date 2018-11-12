import os
import numpy as np
from tqdm import tqdm

from src.optimization.problems.BinaryProblems import OneMax
from src.optimization.problems.BinaryProblems import ZeroMax
from src.optimization.problems.BinaryProblems import KNAPSACK
from src.optimization.metaheuristics.single_objective.BCSO import BCSO


def main():
    os.chdir('../../..')
    num_exec = 30
    num_iter = 500
    dimensions = [10, 15, 20, 23, 50, 100]

    w = 1
    smp = 4
    cdc = 0.1
    pmo = 0.1
    mr = 0.5
    num_particles = 30
    c1 = 1

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

    # BP = OneMax(dim)
    # BP = ZeroMax(dim)
    for dim in tqdm(dimensions):
        BP = KNAPSACK(dim)
        file_path = "results/BCSO/{}/".format(BP.name)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        run_experiments(num_particles, num_iter, w, c1, smp, cdc, pmo, mr, num_exec, BP, file_path)


def run_experiments(swarm_size, n_iter, w, c1, smp, cdc, pmo, mr, num_runs, problem, save_dir):
    alg_name = "BCSO"
    console_out = "Algorithm: {} Function: {} Dimensions: {} Execution: {} Best Cost: {}"
    if save_dir:
        f_handle_cost_iter = file(save_dir + "/BCSO_" + problem.name + "_" + str(problem.dim) + "_dim_cost_iter.txt",
                                  'w+')
        f_handle_cost_eval = file(save_dir + "/BCSO_" + problem.name + "_" + str(problem.dim) + "_dim_cost_eval.txt",
                                  'w+')
    for run in range(num_runs):
        opt1 = BCSO(problem, swarm_size=swarm_size, n_iter=n_iter, w=w, c1=c1, smp=smp, cdc=cdc, pmo=pmo, mr=mr)

        opt1.optimize()
        print console_out.format(alg_name, problem.name, problem.dim, run + 1, opt1.best_cat.cost)

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
