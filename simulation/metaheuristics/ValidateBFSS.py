import os
import numpy as np
from tqdm import tqdm

from src.optimization.problems.BinaryProblems import OneMax
from src.optimization.problems.BinaryProblems import ZeroMax
from src.optimization.problems.BinaryProblems import KNAPSACK
from src.optimization.metaheuristics.single_objective.BFSS import BFSS


def main():
    os.chdir('../../..')

    dimensions = [10, 15, 20, 23, 50, 100]
    num_runs = 30
    n_iter = 500
    school_size = 30
    threshold_individual_initial = 0.9
    threshold_individual_final = 0.4
    threshold_instintive = 0.4
    threshold_volitive_initial = 0.1
    threshold_volitive_final = 0.6
    min_w = 1
    w_scale = n_iter / 2.0

    # BP = OneMax(dim)
    # BP = ZeroMax(dim)
    for dim in tqdm(dimensions):
        BP = KNAPSACK(dim)
        file_path = "results/BFSS/{}/".format(BP.name)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        run_experiments(BP, n_iter, school_size, threshold_individual_initial, threshold_individual_final,
                        threshold_instintive, threshold_volitive_initial, threshold_volitive_final, min_w, w_scale,
                        num_runs, file_path)


def run_experiments(problem, n_iter, school_size, threshold_individual_initial, threshold_individual_final,
                    threshold_instintive, threshold_volitive_initial, threshold_volitive_final, min_w, w_scale,
                    num_runs, save_dir):
    alg_name = "BFSS"
    console_out = "Algorithm: {} Function: {} Dimensions: {} Execution: {} Best Cost: {}"
    if save_dir:
        f_handle_cost_iter = file(save_dir + "/BFSS_" + problem.name + "_" + str(problem.dim) + "_dim_cost_iter.txt",
                                  'w+')
        f_handle_cost_eval = file(save_dir + "/BFSS_" + problem.name + "_" + str(problem.dim) + "_dim_cost_eval.txt",
                                  'w+')

    for run in range(num_runs):
        opt1 = BFSS(problem, n_iter, school_size, threshold_individual_initial, threshold_individual_final,
                    threshold_instintive, threshold_volitive_initial, threshold_volitive_final, min_w, w_scale)

        opt1.optimize()
        print console_out.format(alg_name, problem.name, problem.dim, run + 1, opt1.best_fish.cost)

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
