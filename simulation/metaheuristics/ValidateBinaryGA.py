import os
import numpy as np
from tqdm import tqdm

from src.optimization.problems.BinaryProblems import OneMax
from src.optimization.problems.BinaryProblems import ZeroMax
from src.optimization.problems.BinaryProblems import KNAPSACK
from src.optimization.metaheuristics.single_objective.BinaryGA import BinaryGA


def main():
    os.chdir('../../..')
    num_exec = 30
    pop_size = 30
    num_iter = 500
    dimensions = [10, 15, 20, 23, 50, 100]

    mutation = 0.66
    cross = 0.92

    # BP = OneMax(dim)
    # BP = ZeroMax(dim)

    for dim in tqdm(dimensions):
        BP = KNAPSACK(dim)
        file_path = "results/BGA/{}/".format(BP.name)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        run_experiments(mutation, cross, num_iter, pop_size, num_exec, BP, file_path)


def run_experiments(mutation, cross, num_iter, num_particles, num_runs, problem, save_dir):
    alg_name = "BGA"
    console_out = "Algorithm: {} Function: {} Dimensions: {} Execution: {} Best Cost: {}"
    if save_dir:
        f_handle_cost_iter = file(
            save_dir + "/BGA_" + problem.name + "_" + str(problem.dim) + "_dim_cost_iter.txt", 'w+')
        f_handle_cost_eval = file(
            save_dir + "/BGA_" + problem.name + "_" + str(problem.dim) + "_dim_cost_eval.txt", 'w+')
    for run in range(num_runs):
        opt1 = BinaryGA(problem, pop_size=num_particles, mutation_rate=mutation, cross_rate=cross, max_iter=num_iter)

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


def tune_paramiters():
    dim = 100
    pop_size = 30
    n_iter = 500

    # BP = OneMax(dim)
    # BP = ZeroMax(dim)
    BP = KNAPSACK(dim)

    cost = -1
    mutation_rate = -1
    cross_rate = -1

    def shuffle(bits):
        np.random.shuffle(bits)
        return bits

    # ga = BinaryGA(BP, pop_size=pop_size, mutation_rate=0.668421052632, cross_rate=0.926315789474, max_iter=300)
    # ga.set_cross_operator(np.bitwise_or)
    # ga.optimize()

    for m_rate in np.linspace(0.3, 1, 20):
        for c_rate in np.linspace(0.3, 1, 20):
            ga = BinaryGA(BP, pop_size=pop_size, mutation_rate=m_rate, cross_rate=c_rate, max_iter=300)
            ga.set_cross_operator(np.bitwise_or)
            # ga.set_mutation_operator(shuffle)
            ga.optimize()
            if ga.optimal_solution.cost > cost:
                cost = ga.optimal_solution.cost
                mutation_rate = m_rate
                cross_rate = c_rate

    print('Optimatal mutation_rate: {}'.format(mutation_rate))
    print('Optimatal cross_rate: {}'.format(cross_rate))


if __name__ == '__main__':
    main()
