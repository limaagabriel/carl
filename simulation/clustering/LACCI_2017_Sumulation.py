import os

import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering.swarm.pso.PSOC import PSOC
from src.clustering.swarm.pso.KMPSOC import KMPSOC
from src.clustering.swarm.pso.PSC import PSC
from src.clustering.swarm.pso.PSOCKM import PSOCKM
from src.clustering.tradicional.KMeans import KMeans


from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
import time

from src.clustering.evaluation.Metrics import Metrics


def main(parameters_simulation=None):
    data_file = open(parameters_simulation)
    param_simultations = json.load(data_file)

    pathname_dataset = param_simultations["pathname_dataset"]
    pathname_dir_results = param_simultations["pathname_dir_results"]
    num_simulations = param_simultations["NUN_SIMULATIONS"]  # 30
    num_trials = param_simultations["NUM_TRIALS"]  # 50

    num_iterations = param_simultations["NUM_ITERATIONS"]  # 500
    swarm_size = param_simultations["SWARM_SIZE"]  # 15
    name_classifier = param_simultations["ALGORITHMS"]
    clustering_metrics = param_simultations["EVALUATION_METRICS"]

    criteria = []
    table_criteria = {}

    for idx_alg in range(len(clustering_metrics)):
        value = clustering_metrics[idx_alg]
        criteria.append(value['criteria'])
        table_criteria[value['criteria']] = (clustering_metrics[idx_alg]['name'], clustering_metrics[idx_alg]['file'])

    time.sleep(2)
    pathname_output = pathname_dir_results + '/2017_LA-CCI_ClusteringSimulation'
    currDirectory = (pathname_output + '-ID-' + datetime.now().strftime('%d-%b-%Y-%Hh:%Mm:%Ss'))
    pathname_output = currDirectory

    if not os.path.exists(pathname_output):
        os.makedirs(pathname_output)
    else:
        raise Exception("This simulation cannot execute!")

    time.sleep(1)
    timestamp = datetime.now().strftime('%a, %d %b %Y at %Hh:%Mm:%Ss')
    if not os.path.exists(pathname_output + '/timestamp.txt'):
        file_timestamp = open(pathname_output + '/timestamp.txt', 'a')
        file_timestamp.write('This simulation started on ' + timestamp)
        file_timestamp.close()
        print("This simulation started on " + timestamp)
    else:
        raise Exception("This simulation cannot execute!")

    print("Loading dataset")

    df = pd.read_excel(io=pathname_dataset, sheetname='Plan1')
    df.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    print pathname_dataset

    x = df.iloc[:, :].values.astype(float)
    print("Normalizing dataset so that all dimenions are in the same scale")
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    indices_attributes_4p = np.array([1,7,8,10,12,13,16])
    indices_attributes_4p = indices_attributes_4p - 1

    indices_attributes_5p = np.array([1, 7, 8, 10, 12, 13])
    indices_attributes_5p = indices_attributes_5p - 1

    x = x[:,indices_attributes_4p]



    # x = np.array(x)
    # The term clf means classifier

    # name_classifier = ['PSOC']
    # name_classifier = ['PSC']




    KList = range(2, 10)





    idx_success_simulation = {}
    idx_trial = {}
    idx_fail_simulation = {}
    check_sim_its_over = {}

    for idx_alg in range(len(name_classifier)):
        idx_success_simulation[name_classifier[idx_alg]] = {}
        idx_trial[name_classifier[idx_alg]] = {}
        idx_fail_simulation[name_classifier[idx_alg]] = {}
        check_sim_its_over[name_classifier[idx_alg]] = {}
        for k in KList:
            idx_success_simulation[name_classifier[idx_alg]][k] = 0
            idx_trial[name_classifier[idx_alg]][k] = 0
            idx_fail_simulation[name_classifier[idx_alg]][k] = 0
            check_sim_its_over[name_classifier[idx_alg]][k] = False







    metrics_list_sim_by_algorithm_and_k = {}

    mean_metric_by_algorithm_and_k = {}
    std_metric_by_algorithm_and_k = {}

    for m in range(len(criteria)):
        mean_metric_by_algorithm_and_k[criteria[m]] = {}
        std_metric_by_algorithm_and_k[criteria[m]] = {}

    for m in range(len(criteria)):
        metrics_list_sim_by_algorithm_and_k[criteria[m]] = {}
        for idx_alg in range(len(name_classifier)):
            metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]] = {}
            for k in KList:
                metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][k] = []



    for m in range(len(criteria)):
        min_value_mean = {}
        max_value_mean = {}

        min_value_std = {}
        max_value_std = {}

        for m in range(len(criteria)):
            min_value_mean[criteria[m]] = np.inf
            max_value_mean[criteria[m]] = -np.inf

            min_value_std[criteria[m]] = np.inf
            max_value_std[criteria[m]] = -np.inf


    #First the simulation
    #(idx_success_simulation < num_simulations) and (idx_trial < num_trials)
    finished = False
    idx_fig = 1

    Round = 1

    while not finished:






        list_results_by_metric = {}


        for m in range(len(criteria)):
            list_results_by_metric[criteria[m]] = []


        for idx_alg in range(len(name_classifier)):




            for m in range(len(criteria)):
                mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]] = np.zeros((len(KList),))
                std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]] = np.zeros((len(KList),))





            for k in KList:

                if not check_sim_its_over[name_classifier[idx_alg]][k]:

                    create_dir_algorithms_and_k(idx_alg, k, name_classifier, pathname_output)

                    pathname_output_clusters = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                               + '/clusters/' + str(k) + '-centroids'
                    pathname_output_metrics = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                              + '/metrics/' + str(k) + '-centroids'
                    pathname_output_evolution_success = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                                        + '/evolution/success/' + str(k) + '-centroids'
                    pathname_output_evolution_fail = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                                     + '/evolution/fail/' + str(k) + '-centroids'


                    if (name_classifier[idx_alg] == 'KMPSOC'):
                        clf = KMPSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iterations, w=0.72, c1=1.49, c2=1.49)
                    elif (name_classifier[idx_alg] == 'PSOC'):
                        clf = PSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iterations, w=0.72, c1=1.49, c2=1.49)
                    elif (name_classifier[idx_alg] == 'PSOCKM'):
                        clf = PSOCKM(n_clusters=k, swarm_size=swarm_size, n_iter=num_iterations, w=0.72, c1=1.49, c2=1.49)
                    elif (name_classifier[idx_alg] == 'KMeans'):
                        clf = KMeans(n_clusters=k)
                    elif (name_classifier[idx_alg] == 'PSC'):
                        clf = PSC(swarm_size=k, n_iter=num_iterations, w=0.95, c1=2.05, c2=2.05, c3=1.0, c4=1.0, v_max=0.01)

                    clf.fit(x)

                    centroids = clf.centroids

                    Round = Round + 1

                    if clf.solution.number_of_effective_clusters == k:
                        filename = pathname_output_clusters + '/centroid-' + str(k) + \
                                   '-success-simulation-' + str(
                            idx_success_simulation[name_classifier[idx_alg]][k]  + 1) + '.csv'
                    else:
                        filename = pathname_output_clusters + '/centroid-' + str(k) \
                                   + '-fail-simulation-' + str(idx_fail_simulation[name_classifier[idx_alg]][k]  + 1) + '.csv'

                    dataframe_centroids = pd.DataFrame(centroids)
                    dataframe_centroids.transpose().to_csv(filename, sep=" ", index=False)

                    if clf.solution.number_of_effective_clusters == k:
                        filename = pathname_output_clusters + '/cluster-' + str(k) \
                                   + '-success-simulation-' + str(idx_success_simulation[name_classifier[idx_alg]][k]  + 1) + '.cluster'
                    else:
                        filename = pathname_output_clusters + '/cluster-' + str(k) + \
                                   '-fail-simulation-' + str(idx_fail_simulation[name_classifier[idx_alg]][k]  + 1) + '.cluster'

                    file = open(filename, 'w')
                    file.write(str(len(clf.centroids)) + '\n')
                    file.write(str(clf.solution.number_of_effective_clusters) + '\n')
                    for c in range(len(clf.centroids)):
                        if len(clf.solution.clusters[c]) > 0:
                            file.write(str(len(clf.solution.clusters[c])) + '\n')
                            for xi in range(len(clf.solution.clusters[c])):
                                file.write(str(clf.solution.clusters[c][xi][0]))
                                for xij in range(1, len(clf.solution.clusters[c][xi])):
                                    file.write(' ' + str(clf.solution.clusters[c][xi][xij]))
                                file.write('\n')
                    file.close()

                    if clf.solution.number_of_effective_clusters == k:
                        evol_directory = pathname_output_evolution_success + '/evolution-' + str(k) \
                                         + '-success-simulation-' + str(idx_success_simulation[name_classifier[idx_alg]][k]  + 1)

                        if not os.path.exists(evol_directory):
                            os.makedirs(evol_directory)

                        store_evolution(evol_directory, clf.debugger, k, idx_success_simulation[name_classifier[idx_alg]][k] + 1)

                    else:
                        evol_directory = pathname_output_evolution_fail + '/evolution-' + str(k) + \
                                         '-fail-simulation-' + str(idx_fail_simulation[name_classifier[idx_alg]][k] + 1)

                        if not os.path.exists(evol_directory):
                            os.makedirs(evol_directory)

                        store_evolution(evol_directory, clf.debugger, k, idx_fail_simulation[name_classifier[idx_alg]][k]  + 1)


                    if clf.solution.number_of_effective_clusters == k:

                        for m in range(len(criteria)):
                            value = Metrics.clustering_evaluation(criteria=criteria[m], centroids=centroids, data=x, clf=clf)
                            metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][k].append(value)

                    for m in range(len(criteria)):
                        pd.DataFrame(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][k]).to_csv(
                            pathname_output_metrics + '/' + table_criteria[criteria[m]][1] + '.csv', sep=" ")










                    if clf.solution.number_of_effective_clusters == k:
                        idx_success_simulation[name_classifier[idx_alg]][k] = idx_success_simulation[name_classifier[idx_alg]][k] + 1
                    else:
                        idx_fail_simulation[name_classifier[idx_alg]][k] = idx_fail_simulation[name_classifier[idx_alg]][k] + 1

                    idx_trial[name_classifier[idx_alg]][k] = idx_trial[name_classifier[idx_alg]][k] + 1

                    if (idx_success_simulation[name_classifier[idx_alg]][k] >= num_simulations
                        or  idx_trial[name_classifier[idx_alg]][k] >= num_trials):

                        check_sim_its_over[name_classifier[idx_alg]][k] = True
                        for d in  check_sim_its_over.values():
                            if not all(d.values()):
                                break
                        else:
                            finished = True

                    print("Round(" + str(Round) + ") ........................ " +
                          "(SUCCESS = " + str(idx_success_simulation[name_classifier[idx_alg]][k]) + ", "
                                                                                                       "FAIL = " + str(
                        idx_fail_simulation[name_classifier[idx_alg]][k]) + ", " +
                          "TRAIL = " + str(idx_trial[name_classifier[idx_alg]][k]) + ", " + "CLF = "
                          + name_classifier[idx_alg] + ", K = " + str(k) + ")" + "\n")



            for m in range(len(criteria)):
                idx_k = 0
                for ik in KList:
                    if len(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][ik]) > 0:
                        mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][idx_k] = \
                            np.mean(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][ik])

                        std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][idx_k]  = \
                            np.std(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][ik])

                        list_results_by_metric[criteria[m]].append(
                            [name_classifier[idx_alg], ik, np.mean(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][ik]),
                             np.std(metrics_list_sim_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]][ik])])
                    idx_k = idx_k + 1

            for m in range(len(criteria)):
                min_value_mean[criteria[m]] = np.inf
                max_value_mean[criteria[m]] = -np.inf


                max_value_std[criteria[m]] = -np.inf

            for m in range(len(criteria)):
                min_value_mean[criteria[m]] = np.amin(mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])
                max_value_mean[criteria[m]] = np.amax(mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])

                max_value_std[criteria[m]] = np.amax(std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])




            for m in range(len(criteria)):
                #print mean_metric_by_algorithm_and_k
                plt.figure()
                plt.title(str(name_classifier[idx_alg]) + ' - ' + table_criteria[criteria[m]][0])
                # plt.errorbar(KList, mean_metric_by_algorithm_and_k[criteria[m]], yerr=std_metric_by_algorithm_and_k[criteria[m]], linewidth=0.5, elinewidth=0.5, color='b', capthick=2, barsabove=True)
                plt.errorbar(KList, mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]],
                             yerr=(std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]]), linewidth=0.5, elinewidth=0.5,
                             color='b')
                plt.plot(KList, mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]], color='b', marker='o',
                         linewidth=0.5,
                         markersize=5)
                plt.xlabel('K')
                plt.ylabel(table_criteria[criteria[m]][0])
                ymin = min_value_mean[criteria[m]] - max_value_std[criteria[m]]
                ymax = max_value_mean[criteria[m]] + max_value_std[criteria[m]]
                delta = ymax - ymin
                plt.ylim([ymin-0.01*delta, ymax+0.01*delta])
                plt.tight_layout()
                plt.savefig(pathname_output + '/data/' + name_classifier[idx_alg] + "/" + name_classifier[
                    idx_alg] + "-" +
                            table_criteria[criteria[m]][1] + '.pdf')
                plt.close("all")

        for m in range(len(criteria)):
            df_by_metric = pd.DataFrame(list_results_by_metric[criteria[m]])
            df_by_metric.columns = ['ALGORITHM', 'CLUSTERS', 'MEAN', 'STD']
            df_by_metric.to_csv(
                pathname_output + '/data/' + table_criteria[criteria[m]][1] + '.csv',
                index=False)



    # mean_metric_by_algorithm[name_classifier[idx_alg]] = mean_metric_by_algorithm_and_k
    # std_metric_by_algorithm[name_classifier[idx_alg]] = std_metric_by_algorithm_and_k
    #


    for m in range(len(criteria)):
        min_value_mean[criteria[m]] = np.inf
        max_value_mean[criteria[m]] = -np.inf


        max_value_std[criteria[m]] = -np.inf

    for m in range(len(criteria)):
        for idx_alg in range(len(name_classifier)):
            curr_min_mean = np.amin(mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])
            curr_max_mean = np.amax(mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])
            min_value_mean[criteria[m]] = np.minimum(min_value_mean[criteria[m]], curr_min_mean)
            max_value_mean[criteria[m]] = np.maximum(max_value_mean[criteria[m]], curr_max_mean)


            curr_max_std = np.amax(std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]])

            max_value_std[criteria[m]] = np.maximum(max_value_std[criteria[m]], curr_max_std)



    for m in range(len(criteria)):
        for idx_alg in range(len(name_classifier)):
            plt.figure()
            plt.title(str(name_classifier[idx_alg]) + ' - ' + table_criteria[criteria[m]][0])
            # plt.errorbar(KList, mean_metric_by_algorithm_and_k[criteria[m]], yerr=std_metric_by_algorithm_and_k[criteria[m]], linewidth=0.5, elinewidth=0.5, color='b', capthick=2, barsabove=True)
            plt.errorbar(KList, mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]],
                         yerr=std_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]], linewidth=0.5, elinewidth=0.5, color='b')
            plt.plot(KList, mean_metric_by_algorithm_and_k[criteria[m]][name_classifier[idx_alg]], color='b', marker='o', linewidth=0.5,
                     markersize=5)
            plt.xlabel('K')
            plt.ylabel(table_criteria[criteria[m]][0])
            ymin = min_value_mean[criteria[m]] - max_value_std[criteria[m]]
            ymax = max_value_mean[criteria[m]] + max_value_std[criteria[m]]
            delta = ymax - ymin
            plt.ylim([ymin-0.01*delta , ymax + 0.01*delta])
            plt.tight_layout()
            plt.savefig(pathname_output + '/data/' + name_classifier[idx_alg] + "/" + name_classifier[idx_alg] + "-" +
                        table_criteria[criteria[m]][1] + '_final.pdf')
            plt.close("all")





def create_dir_algorithms_and_k(idx_alg, k, name_classifier, pathname_output):
    pathname_output_clusters = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                               + '/clusters/' + str(k) + '-centroids'
    if not os.path.exists(pathname_output_clusters):
        os.makedirs(pathname_output_clusters)
    pathname_output_metrics = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                              + '/metrics/' + str(k) + '-centroids'
    if not os.path.exists(pathname_output_metrics):
        os.makedirs(pathname_output_metrics)
    pathname_output_evolution_success = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                        + '/evolution/success/' + str(k) + '-centroids'
    pathname_output_evolution_fail = pathname_output + '/data/' + name_classifier[idx_alg] + '/' \
                                     + '/evolution/fail/' + str(k) + '-centroids'
    if not os.path.exists(pathname_output_evolution_success):
        os.makedirs(pathname_output_evolution_success)
    if not os.path.exists(pathname_output_evolution_fail):
        os.makedirs(pathname_output_evolution_fail)


def store_evolution(evol_directory, debugger, k, idx_sim):
    rep_best_centroids = debugger.rep_best_centroids
    rep_best_cost = debugger.rep_best_cost
    rep_iteration = debugger.rep_iteration

    pathname = evol_directory + '/evolution_cost-' + str(k) + '-' + str(idx_sim) + '.csv'
    pd.DataFrame(rep_best_cost).to_csv(pathname)

    # for i in range(len(rep_iteration)):
    #     pathname = evol_directory + '/centroid-' + str(k) + '-' + str(idx_sim) + '-iter-' + str(rep_iteration[i]) + '.csv'
    #     centroid = rep_best_centroids[i]
    #     pd.DataFrame(centroid).to_csv(pathname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2017 LA-CCI Simulation')

    parser.add_argument('--file', metavar='path', required=True,
                        help='the path to file json')

    args = parser.parse_args()
    main(args.file)
