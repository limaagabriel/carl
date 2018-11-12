import os
import numpy as np
from tqdm import tqdm

from src.optimization.problems.BinaryProblems import OneMax
from src.optimization.problems.BinaryProblems import ZeroMax
from src.optimization.problems.BinaryProblems import KNAPSACK
from src.optimization.metaheuristics.single_objective.BPSO import BPSO


def main():
    os.chdir('../../..')
    num_exec = 30
    pop_size = 30
    num_iter = 500
    dimensions = [10, 15, 20, 23, 50, 100]

    lb_w = 0.1
    up_w = 0.9
    c1 = (0.72984 * 2.05)
    c2 = (0.72984 * 2.05)

    for dim in tqdm(dimensions):
        BP = KNAPSACK(dim)
        file_path = "results/BPSO/{}/".format(BP.name)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        run_experiments(lb_w, up_w, c1, c2, num_iter, pop_size, num_exec, BP, file_path)


def run_experiments(lb_w, up_w, c1, c2, num_iter, num_particles, num_runs, problem, save_dir):
    alg_name = "BPSO"
    console_out = "Algorithm: {} Function: {} Dimensions: {} Execution: {} Best Cost: {}"
    if save_dir:
        f_handle_cost_iter = file(
            save_dir + "/BPSO_" + problem.name + "_" + str(problem.dim) + "_dim_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(
            save_dir + "/BPSO_" + problem.name + "_" + str(problem.dim) + "_dim_cost_eval.txt", 'w+')
    for run in range(num_runs):
        opt1 = BPSO(fitness=problem, pop_size=num_particles, max_iter=num_iter, lb_w=lb_w, up_w=up_w, c1=c1, c2=c2,
                    v_max=100000)

        opt1.optimize()
        print console_out.format(alg_name, problem.name, problem.dim, run + 1, opt1.optimal_solution.cost)

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
