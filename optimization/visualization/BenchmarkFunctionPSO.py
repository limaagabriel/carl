import matplotlib.pyplot as plt
import numpy as np

def plotGraphics(config):
    txt_files = []

    function_names = config.keys()

    for i in range(len(function_names)):

        optimum_cost_tracking_foreach_run = np.loadtxt(config[function_names[i]])

        TRESHOLD = (10 ** -8)

        mean_optimum_cost_tracking = np.mean(optimum_cost_tracking_foreach_run, axis=0)

        mean_optimum_cost_tracking[mean_optimum_cost_tracking < TRESHOLD] = TRESHOLD

        plt.plot([(j + 1) for j in range(len(mean_optimum_cost_tracking))], mean_optimum_cost_tracking, c='b')
        plt.legend([function_names[i]])
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.yscale('log')
        plt.show()

def main():
    PATHNAME = 'C:/Users/Elliackin Figueiredo/Elliackin Figueiredo HOME' \
               '/PosDocResearchSwarmBasedClustering/SRCSwarmClustering/'

    pathname = PATHNAME + 'PSOSimulation/'

    functions = ['Rosenbrock', 'Rastrigin', 'Schwefel', 'Griewank',
              'Ackley', 'Sphere']

    config = {}

    for i in range(len(functions)):
        path = pathname + 'simulation_' + functions[i] + '_pso.txt'
        config[functions[i]] = path

    plotGraphics(config)

if __name__ == '__main__':
    main()
