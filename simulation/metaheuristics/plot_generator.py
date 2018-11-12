import os
import matplotlib.pyplot as plt
from src.simulation.metaheuristics.Utils import generate_lines_plot, generate_box_plots, generate_stats_test_results


def plot_generator(files_loc, functions, algorithms, dimensions, num_iter, num_eval=None, lines_plot=True,
                   box_plot=True, wilcox=True, friedman=True, grouped=True):
    if lines_plot:
        print("Generating Lines plot")
        if num_eval:
            generate_lines_plot(files_loc=files_loc, functions=functions, algorithms=algorithms,
                                dimensions=dimensions, num_iter=num_eval, per_eval=True, grouped=grouped, n_exec=30)
        if num_iter:
            generate_lines_plot(files_loc=files_loc, functions=functions, algorithms=algorithms, num_iter=num_iter,
                                dimensions=dimensions, per_eval=False, grouped=grouped, n_exec=30)

    if box_plot:
        print("Generating Box plot")
        if num_eval:
            generate_box_plots(files_loc=files_loc, functions=functions, algorithms=algorithms,
                               dimensions=dimensions, num_iter=num_eval, per_eval=True, grouped=grouped)
        if num_iter:
            generate_box_plots(files_loc=files_loc, functions=functions, algorithms=algorithms,
                               dimensions=dimensions, num_iter=num_iter, per_eval=False, grouped=grouped)

    if any([wilcox, friedman]):
        print("Applying Tests")
        if num_eval:
            generate_stats_test_results(files_loc=files_loc, functions=functions, algorithms=algorithms,
                                        dimensions=dimensions, num_iter=num_eval,  wilcox=wilcox, per_eval=False)
        if num_iter:
            generate_stats_test_results(files_loc=files_loc, functions=functions, algorithms=algorithms,
                                        dimensions=dimensions, num_iter=num_iter,  wilcox=wilcox, per_eval=False)

    if any([lines_plot, box_plot, wilcox]):
        plt.show()


def main():
    os.chdir('../../..')
    os.getcwd()
    files_loc = os.getcwd() + "/results/cat_results"
    functions = ["Knapsack"]
    algorithms = ["BCSO", "DSBPSO", "BBCSO", "BFSS", "BPSO", "BGA"]  #, "BBFO","BooleanPSO"]
    dimensions = ["10", "15", "20", "23", "50", "100"]
    # dimensions = ["23"]
    num_iter = 500
    num_eval = None

    plot_generator(files_loc=files_loc, functions=functions, dimensions=dimensions, algorithms=algorithms,
                   num_iter=num_iter, num_eval=num_eval, lines_plot=True, box_plot=True, wilcox=False, friedman=False,
                   grouped=False)


if __name__ == '__main__':
    main()
