import os
import scipy.stats
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt


def show_progress_bar(per):
    CURSOR_UP_ONE_ERASE = '\x1b[1A\x1b[2K\x1b[1A'
    prog = "Progress: {:10.2f}%".format(per * 100.0)
    print(CURSOR_UP_ONE_ERASE)
    print(prog)


def create_dir(path):
    directory = os.path.dirname(path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


def swarm_evolution(swarm):
    xar = []
    yar = []
    s = []

    for ind in swarm:
        xar.append(ind.pos[0])
        yar.append(ind.pos[1])
        s.append(ind.weight)

    plt.ion()
    plt.clf()
    plt.scatter(xar, yar, s=s)
    plt.ylim([-100, 100])
    plt.xlim([-100, 100])
    plt.pause(0.05)


def generate_box_plots(files_loc, functions, algorithms, dimensions, num_iter, per_eval, grouped=True):
    plt.figure()
    title = "{} function {} dimensions"
    if per_eval:
        file_ = "_dim_cost_eval.txt"
    else:
        file_ = "_dim_cost_iter.txt"

    for idx in range(len(functions)):
        for dim in range(len(dimensions)):
            if grouped:
                plt.subplot(np.ceil(len(dimensions) / 3.0), 3, dim + 1)
            else:
                plt.figure()
            box = []
            for alg in algorithms:
                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                f = open(file_name, 'r')
                execs = []
                for line in f:
                    data = line.split()

                    data = [data[i] for i in range(num_iter)]
                    data = np.asarray(data, dtype=np.float)
                    execs.append(data[num_iter - 1])
                box.append(execs)

            plt.boxplot(box)
            # plt.title(title.format(functions[idx], dimensions[dim]))
            plt.xlabel('Heuristic')
            plt.ylabel('Fitness')
            plt.xticks(range(1, len(algorithms) + 1), algorithms)
            # plt.yscale('log')
            output_name_w = "_box_plots"
            plt.savefig(os.path.join(files_loc, functions[idx] + "_" + str(dimensions[dim]) + output_name_w))


def generate_lines_plot(files_loc, functions, algorithms, dimensions, num_iter, per_eval, grouped, n_exec=30):
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    plt.figure()
    title = "{} function {} dimensions"
    if per_eval:
        file_ = "_dim_cost_eval.txt"
        x_label = 'Evaluations'
    else:
        file_ = "_dim_cost_iter.txt"
        x_label = 'Iterations'

    for idx in range(len(functions)):
        for dim in range(len(dimensions)):
            if grouped is True:
                plt.subplot(np.ceil(len(dimensions) / 3.0), 3, dim + 1)
            else:
                plt.figure()

            for alg in algorithms:
                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                f = open(file_name, 'r')
                count = 0
                lines = []
                for line in f:
                    data = line.split()
                    if count == 0:
                        lines = data
                    elif count < n_exec:
                        lines = data + lines
                    count += 1
                    if count == n_exec:
                        lines = np.asarray(lines, dtype=np.float)
                        plt.plot([lines[i] for i in range(num_iter)], linestyle=next(linecycler))
                        plt.yscale('log')
                        count = 0

            # plt.title(title.format(functions[idx], dimensions[dim]))
            plt.legend(algorithms)
            plt.xlabel(x_label)
            plt.ylabel('Fitness')
            plt.yticks([9.8e+3, 9.9e+3, 1.0e+4])
            # plt.yscale('log')
            output_name_w = "_lines_plot"
            # plt.savefig(os.path.join(files_loc, functions[idx] + "_" + str(dimensions[dim]) + output_name_w))


def generate_stats_test_results(files_loc, functions, algorithms, dimensions, num_iter, wilcox, per_eval):
    if per_eval:
        file_ = "_dim_cost_eval.txt"
        output_name_w = "_dim_wicoxon_test_results_eval.csv"
    else:
        file_ = "_dim_cost_iter.txt"
        output_name_w = "_dim_wicoxon_test_results_iter.csv"

    for idx in range(len(functions)):
        for dim in range(len(dimensions)):
            populations = []
            for alg in algorithms:
                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                f = open(file_name, 'r')
                execs = []
                for line in f:
                    data = line.split()
                    data = np.asarray(data, dtype=np.float)
                    execs.append(data[num_iter-1])
                populations.append(execs)

            for alg1 in range(len(populations)):
                matrix = []
                for alg2 in range(len(populations)):
                    test = scipy.stats.ranksums(populations[alg1], populations[alg2])
                    if test[1] < 0.05:
                        if test[0] > 0:
                            result = "ganha do"
                        else:
                            result = "perde do"
                    else:
                        result = "empata com"
                    matrix.append(
                        ["O algoritm {} {} {}".format(algorithms[alg1], result, algorithms[alg2])])

                df = pd.DataFrame(matrix)
                new_header = df.iloc[0]
                df = df[1:]
                df = df.rename(columns=new_header)
                df.to_csv(os.path.join(files_loc, functions[idx] + "_" + algorithms[alg1] + "_" + str(dimensions[dim]) + output_name_w), index=False)

