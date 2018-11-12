import os
import copy
import operator
import community
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import plotly.plotly as py
import graph_tool.all as gt
import matplotlib.pyplot as plt
from BeautifulSoup import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler


def get_best_simulation(file_name, num_exec=30):
    centroids_matrix = []
    for sim in range(0, num_exec):
        name = file_name.format(sim)
        centroids_ = {}
        temp_centroid = pd.read_csv(os.path.join(name), sep=",")
        temp_centroid = temp_centroid.values
        for c in range(len(temp_centroid)):
            np.delete(temp_centroid[c], 0)
            centroids_[c] = np.delete(temp_centroid[c], 0)
        centroids_matrix.append(sorted(centroids_.items(), key=operator.itemgetter(0)))

    num_centroids = len(centroids_matrix[0])
    dist = np.zeros((num_exec, num_centroids))

    for sim in range(num_exec):
        for cen_num in range(num_centroids):
            centroid = (centroids_matrix[sim][cen_num])[1]
            dist[sim][cen_num] = np.linalg.norm(centroid)

    med_dist = dist.mean(0)
    dist_to_med = [np.linalg.norm(centroid - med_dist) for centroid in dist]
    minimum_sim = np.argmin(dist_to_med)
    return minimum_sim


def load_data(path, datasets, algorithm, k):
    normalized_dfs = []
    centroids = []

    place_names_h = pd.read_csv(os.path.join(path, "cities_states_names_new.csv"), sep=",")
    place_names_h.head()
    place_names = place_names_h[place_names_h.apply(lambda x: sum([x_ == '?' for x_ in x]) == 0, axis=1)]
    place_names = place_names.iloc[:, :].values.astype(str)

    for key, value in datasets.iteritems():
        # Creating normalized dataset array
        temp_data = pd.read_csv(os.path.join(path, value + ".csv"), sep=";")
        temp_data = temp_data[temp_data.apply(lambda x: sum([x_ == '?' for x_ in x]) == 0, axis=1)]
        temp_data = temp_data.iloc[:, :].values.astype(float)
        std = MinMaxScaler()
        normalized_dfs.append(std.fit_transform(temp_data))

        # Creating centroids array
        ds_num = "dataset_" + str(key)
        alg_name = "algorithm_" + algorithm
        f_name = ds_num + "_" + alg_name + "_k_" + str(k[key]) + "_simulation_{}_centroids.csv"
        best_sim = get_best_simulation(os.path.join(path, ds_num, alg_name, f_name))

        temp_centroid = pd.read_csv(os.path.join(path, ds_num, alg_name, f_name.format(best_sim)), sep=",")
        temp_centroid = temp_centroid.values
        centroids_ = {}
        for c in range(len(temp_centroid)):
            np.delete(temp_centroid[c], 0)
            centroids_[c] = np.delete(temp_centroid[c], 0)
        centroids.append(centroids_)

    return [normalized_dfs, centroids, place_names, place_names_h]


def connect_cities(matrix, cluster):
    items = copy.deepcopy(cluster)
    for item1 in cluster:
        items.remove(item1)
        for item2 in items:
            matrix[item1][item2] += 1
            matrix[item2][item1] += 1
    return matrix


def connect_states(matrix, cluster, names):
    items = copy.deepcopy(cluster)
    for item1 in cluster:
        items.remove(item1)
        for item2 in items:
            try:
                x = int((names[item1])[1])
                y = int((names[item2])[1])

                matrix[x][y] += 1
                matrix[y][x] += 1
            except:
                pass
    return matrix


def connect_cities_states(matrix, cluster, names, same=True):
    items = copy.deepcopy(cluster)
    for item1 in cluster:
        items.remove(item1)
        for item2 in items:
            try:
                state1 = int((names[item1])[1])
                state2 = int((names[item2])[1])
                if (same and (state1 == state2)) or (not same and (state1 != state2)):
                    matrix[item1][item2] += 1
                    matrix[item2][item1] += 1
            except:
                pass
    return matrix


def get_clusters(dataset, centroids):
    count = 0
    clusters = {}
    for c in centroids:
        clusters[c] = set()

    for x in dataset:
        dist = [np.linalg.norm(x - centroids[c]) for c in centroids]
        class_ = dist.index(min(dist))
        clusters[class_].add(count)
        count += 1
    return clusters


def get_statistics(graph, matrix):
    d = np.sum(matrix, axis=1)
    e = np.sum(d) / graph.order()
    print("Average degree: ", e)

    hist = nx.degree_histogram(graph)
    entropy = 0
    for i in range(len(hist)):
        if hist[i] != 0:
            entropy -= hist[i] * np.log2(hist[i])

    print("Entropy: ", entropy)

    average_sp = nx.average_shortest_path_length(graph)
    print("Average shortest path length: ", average_sp)

    average_c = nx.average_clustering(graph)
    print("Network Clustering: ", average_c)

    a_s = nx.degree_assortativity_coefficient(graph)
    print("Assortativeness: ", a_s)

    density = nx.density(graph)
    print("Density:", density)

    plt.bar(range(len(hist)), hist)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.show()

    autovalores = nx.laplacian_spectrum(graph)
    plt.bar(range(len(autovalores)), autovalores)
    plt.title("Autovalues")
    plt.show()


def groups_to_csv(out_dir, groups, type, graph=None):
    res = []
    if type in ["label", "infomap"]:
        res = np.asarray(groups)
    elif type == "louvain":
        temp = np.asarray([groups.get(node) for node in graph.nodes()])
        group = np.max(temp)
        while group >= 0:
            hb = np.argwhere(temp == group)
            hb = [x[0] for x in hb]
            if len(hb) > 0:
                res.append(hb)
            group -= 1
        res = np.asarray(res)
    elif type == "stochastic":
        temp = np.asarray(groups.get_blocks().a)
        group = np.max(temp)
        while group >= 0:
            hb = np.argwhere(temp == group)
            hb = [x[0] for x in hb]
            if len(hb) > 0:
                res.append(hb)
            group -= 1
        res = np.asarray(res)
    else:
        raise Exception("Unknown grouping type")

    f = open(os.path.join(out_dir, type + ".csv"), 'w+')
    for idx in range(len(res)):
        line = ""
        for node in res[idx]:
            if len(line) == 0:
                line = str(node)
            else:
                line += "," + str(node)
        f.write(line + '\n')
    f.close()
    return res


def apply_community_detection(graph, names, output_loc):
    # convert the graph from Networkx format to Graph-tool format.
    # TODO: VERIFY IS IS NEEDED TO REMOVE ISOLATED COMPONENTS
    # singletons = nx.isolates(graph)
    # graph.remove_nodes_from(singletons)
    gt_graph = gt.Graph(directed=False)

    for i in range(graph.number_of_nodes()):
        gt_graph.add_vertex()
    for edge in graph.edges():
        gt_graph.add_edge(gt_graph.vertex(edge[0]), gt_graph.vertex(edge[1]))

    # Draws original graph
    gt.graph_draw(gt_graph, vertex_text=gt_graph.vertex_index, output=os.path.join(output_loc, "original_graph.pdf"),
                  output_size=(5000, 5000), edge_pen_width=1.2)

    # convert the graph from Networkx format iGraph format.
    ig_graph = ig.Graph(1)
    n_graph = ig_graph.Read_Pajek(os.path.join(output_loc, "graph.pajek"))
    # TODO: VERIFY IS IS NEEDED TO REMOVE ISOLATED COMPONENTS
    # singletons = n_graph.vs.select(_degree=0)
    # n_graph.delete_vertices(singletons)

    # Detect communities using Stochastic Block Model technique
    stochastic_groups = gt.minimize_blockmodel_dl(gt_graph, deg_corr=False)
    stochastic_groups = groups_to_csv(output_loc, stochastic_groups, "stochastic")

    # Draws Stochastic Block graph
    state = gt.minimize_nested_blockmodel_dl(gt_graph, deg_corr=False)
    gt.draw_hierarchy(state, output=os.path.join(output_loc, "community_stochastic.pdf"))

    # Detect communities using the Louvain heuristic.
    louvain_groups = community.best_partition(graph)
    values = [louvain_groups.get(node) for node in graph.nodes()]
    louvain_groups = groups_to_csv(output_loc, louvain_groups, "louvain", graph)

    # Draws Louvain graph
    nx.draw_networkx(graph, cmap=plt.get_cmap('jet'), node_color=values, node_size=30, with_labels=True, width=0.01)
    plt.savefig(os.path.join(output_loc, "community_louvain_pdf"), dpi=300)

    # Detect communities using Infomap approach
    infomap_groups = n_graph.community_infomap()
    infomap_groups = groups_to_csv(output_loc, infomap_groups, "infomap")

    # Detect communities using Label Propagation method
    label_propagation_groups = n_graph.community_label_propagation()
    label_propagation_groups = groups_to_csv(output_loc, label_propagation_groups, "label")

    return [stochastic_groups, louvain_groups, infomap_groups, label_propagation_groups]


def get_group_by_index(index, groups):
    for group in range(len(groups)):
        found = np.argwhere(np.asarray(groups[group]) == index)
        found = [x[0] for x in found]
        if len(found) > 0:
            return group
    else:
        return -1


def get_group_by_city_name(name, city_list, groups):
    city_state = str(name).split(',')
    for idx in range(len(city_list)):
        temp_city = city_list[idx]

        if (city_state[0] in temp_city[3]) and (temp_city[0] in city_state[1]):
            return get_group_by_index(idx, groups)
    else:
        return -1


def generate_maps_plot(output_dir, city_names, groups, method):
    # Load the SVG map
    svg = open('data/network_science_project/USA_Counties_with_FIPS_and_names.svg', 'r').read()

    # Load into Beautiful Soup
    soup = BeautifulSoup(svg, selfClosingTags=['defs', 'sodipodi:namedview'])

    # Find counties
    paths = soup.findAll('path')

    # Map colors
    colors = ["#F1EEF6", "#1976D2", "C2185B", "#9E9E9E", "#FFC107", "#CDDC39", "#7B1FA2", "#D32F2F", "#607D8B",
              "#795548", "#512DA8", "#FF5722", "#303F9F", "#0097A7", "#4CAF50", "#DCEDC8", "#673AB7", "#00796B",
              "#B2DFDB", "#FFF9C4", "#E1BEE7", "#DCEDC8", "#C5CAE9", "#FFE0B2", "#607D8B", "#B3E5FC", "#FFA000"]

    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-' \
                 'miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    # Color the counties based on unemployment rat
    for p in paths:
        if p['id'] not in ["State_Lines", "separator"]:
            rate = get_group_by_city_name(p['inkscape:label'], city_names, groups)
            if rate == -1:
                color_class = 0
            else:
                color_class = (rate + 1) % len(colors)

            color = colors[color_class]
            p['style'] = path_style + color

    map_plot = soup.prettify("utf-8")
    with open(os.path.join(output_dir, method + "_communities_map.svg"), "w") as f:
        f.write(str(map_plot))


def calculate_conductance(communites, matrix_, output_dir, method_name):
    conductance = []

    for idx in range(len(communites)):
        intra_val = extra_val = inter_val = 0.0

        intra_nodes = copy.deepcopy(communites[idx])

        extra_temp = copy.deepcopy(communites)
        extra_temp = np.delete(extra_temp, idx)
        extra_nodes = np.asarray([])
        for comm in extra_temp:
            extra_nodes = np.append(extra_nodes, comm)

        # Calculate intra-conductance
        for i in intra_nodes:
            for j in intra_nodes:
                intra_val += matrix_[int(i)][int(j)]

        # Calculate extra-conductance
        for i in extra_nodes:
            for j in extra_nodes:
                extra_val += matrix_[int(i)][int(j)]

        # Calculate inter-conductance
        for i in intra_nodes:
            for j in extra_nodes:
                inter_val += matrix_[int(i)][int(j)]

        if (intra_val > extra_val) and (extra_val > 0):
            conductance.append(inter_val / extra_val)
        if (intra_val < extra_val) and (intra_val > 0):
            conductance.append(inter_val / intra_val)
        else:
            conductance.append(0.0)

    f = open(os.path.join(output_dir, method_name + "_stats.csv"), 'w+')
    f.write("Num of communites," + str(len(communites)) + '\n')
    f.write("Mean Conductance," + str(np.mean(conductance)) + '\n')
    f.write("STD Conductance," + str(np.std(conductance)) + '\n')
    f.close()


def save_community(communities, matrix_, place_names_w_h, out_dir, method):
    for idx in range(len(communities)):
        for city in communities[idx]:
            place_names_w_h['cluster'][city] = idx
            place_names_w_h['degree'][city] = np.sum(matrix_[idx])

    place_names_w_h.to_csv(os.path.join(out_dir, method + '_dataset_communities.csv'), sep=',')

    colors = ["rgb(0,116,217)", "rgb(255,65,54)", "rgb(133,20,75)", "rgb(255,133,27)", "rgb(0, 0, 255)",
              "rgb(255, 255, 0)", "rgb(204, 0, 204)", "rgb(0, 204, 0)", "rgb(255, 51, 0)", "rgb(102, 0, 102)",
              "rgb(102, 51, 0)", "rgb(0, 153, 51)", "rgb(255, 153, 0)", "lightgrey"]
    cities = []
    scale = 100

    for idx in range(len(communities)):
        temp_df = place_names_w_h[(place_names_w_h.cluster == idx)]
        cluster = np.array(temp_df['cluster'])
        cluster.astype(int)
        city = dict(
            type='scattergeo',
            locationmode='USA-states',
            lon=temp_df['lon'],
            lat=temp_df['lat'],
            text=temp_df['city'],
            marker=dict(
                size=temp_df['degree'] / scale,
                color=colors[idx],
                line=dict(width=0.5, color='rgb(40,40,40)'),
                sizemode='area'
            ),
            name='Community - {}'.format(idx))
        cities.append(city)

    layout = dict(
        title='Detected Communities - ' + method,
        showlegend=True,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showland=True,
            landcolor='rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

    fig = dict(data=cities, layout=layout)
    py.iplot(fig, validate=False, filename='Detected Communities_' + method)


def calculate_rank_stability(communites, matrix_, output_dir, method_name):
    stabilities = []

    for idx in range(len(communites)):
        comm_size = len(communites[idx])
        temp_stab = []
        for j in range(len(communites[idx])):
            freq = 0.0
            for i in range(len(communites[idx])):
                if matrix_[int(i)][int(j)] != 0:
                    freq += 1.0
            prob = (freq / comm_size)
            if prob != 0:
                temp_stab.append(prob * np.log2(prob))
            else:
                temp_stab.append(0.0)
        stabilities.append(-1 * np.sum(temp_stab))

    f = open(os.path.join(output_dir, method_name + "_stats.csv"), 'a+')
    f.write("Mean Rank stability," + str(np.mean(stabilities)) + '\n')
    f.write("STD Rank stability," + str(np.std(stabilities)) + '\n')
    f.close()


def run_all_cities(num_cities, place_names, place_names_w_h, normalized_dfs, centroids_list, clusters, output_dir):
    # create graph in which the cities are the nodes
    out_dir = os.path.join(output_dir, "all_cities")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    matrix_ = np.zeros((num_cities, num_cities))
    for i in range(len(normalized_dfs)):
        for c in centroids_list[i]:
            matrix_ = connect_cities(matrix_, ((clusters[i])[c]))
    matrix_cities = np.matrix(matrix_)

    np.savetxt(os.path.join(out_dir, "matrix.csv"), matrix_cities, delimiter=",")
    graph = nx.from_numpy_matrix(matrix_cities)
    nx.write_pajek(graph, os.path.join(out_dir, "graph.pajek"))

    communities = apply_community_detection(graph, place_names, out_dir)
    methods_ = ["stochastic", "louvain", "infomap", "label"]
    for idx in range(len(communities)):
        save_community(communities[idx], matrix_, place_names_w_h, out_dir, methods_[idx])
        calculate_conductance(communities[idx], matrix_, out_dir, methods_[idx])
        calculate_rank_stability(communities[idx], matrix_, out_dir, methods_[idx])
        generate_maps_plot(out_dir, place_names, communities[idx], methods_[idx])


def run_only_states(num_states, normalized_dfs, centroids_list, clusters, place_names, output_dir):
    # create graph in which the states are the nodes
    out_dir = os.path.join(output_dir, "only_states")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    matrix_ = np.zeros((num_states, num_states))
    for i in range(len(normalized_dfs)):
        for c in centroids_list[i]:
            matrix_ = connect_states(matrix_, ((clusters[i])[c]), place_names)
    matrix_states = np.matrix(matrix_)

    np.savetxt(os.path.join(out_dir, "matrix.csv"), matrix_states, delimiter=",")
    graph = nx.from_numpy_matrix(matrix_states)
    nx.write_pajek(graph, os.path.join(out_dir, "graph.pajek"))

    communities = apply_community_detection(graph, out_dir)
    methods_ = ["stochastic", "louvain", "infomap", "label"]
    for idx in range(len(communities)):
        calculate_conductance(communities[idx], matrix_, out_dir, methods_[idx])
        calculate_rank_stability(communities[idx], matrix_, out_dir, methods_[idx])
        generate_maps_plot(out_dir, place_names, communities[idx], methods_[idx])


def run_cities_same_state(num_cities, normalized_dfs, centroids_list, clusters, place_names, output_dir):
    # create graph in which cities from the same state are connected
    out_dir = os.path.join(output_dir, "cities_same_state")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    matrix_ = np.zeros((num_cities, num_cities))
    for i in range(len(normalized_dfs)):
        for c in centroids_list[i]:
            matrix_ = connect_cities_states(matrix_, ((clusters[i])[c]), place_names, True)
    matrix_cities_states = np.matrix(matrix_)

    np.savetxt(os.path.join(out_dir, "matrix.csv"), matrix_cities_states, delimiter=",")
    graph = nx.from_numpy_matrix(matrix_cities_states)
    nx.write_pajek(graph, os.path.join(out_dir, "graph.pajek"))

    communities = apply_community_detection(graph, out_dir)
    methods_ = ["stochastic", "louvain", "infomap", "label"]
    for idx in range(len(communities)):
        calculate_conductance(communities[idx], matrix_, out_dir, methods_[idx])
        calculate_rank_stability(communities[idx], matrix_, out_dir, methods_[idx])
        generate_maps_plot(out_dir, place_names, communities[idx], methods_[idx])


def run_cities_different_states(num_cities, normalized_dfs, centroids_list, clusters, place_names, output_dir):
    # create graph in which cities from the same state are not connected
    out_dir = os.path.join(output_dir, "cities_different_states")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    matrix_ = np.zeros((num_cities, num_cities))
    for i in range(len(normalized_dfs)):
        for c in centroids_list[i]:
            matrix_ = connect_cities_states(matrix_, ((clusters[i])[c]), place_names, False)
    matrix_cities_not_states = np.matrix(matrix_)

    np.savetxt(os.path.join(out_dir, "matrix.csv"), matrix_cities_not_states, delimiter=",")
    graph = nx.from_numpy_matrix(matrix_cities_not_states)
    nx.write_pajek(graph, os.path.join(out_dir, "graph.pajek"))

    communities = apply_community_detection(graph, out_dir)
    methods_ = ["stochastic", "louvain", "infomap", "label"]
    for idx in range(len(communities)):
        calculate_conductance(communities[idx], matrix_, out_dir, methods_[idx])
        calculate_rank_stability(communities[idx], matrix_, out_dir, methods_[idx])
        generate_maps_plot(out_dir, place_names, communities[idx], methods_[idx])


def main():
    num_cities = 499
    num_states = 49
    num_clusters = [4, 3, 5, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3]
    algorithm = "FCMeans"
    os.chdir('../../../..')

    files_loc = "data/network_science_project"
    output_dir = os.path.join(files_loc, "analysis_results")
    datasets = {0: "arthritis", 1: "asthma", 2: "blood", 3: "cancer", 4: "coronary_heart_disease", 5: "diabetes",
                7: "kidney", 8: "mental_health", 9: "physical_health", 10: "pulmonary", 11: "stroke", 12: "teeth_lost"}

    res = load_data(files_loc, datasets, algorithm, num_clusters)
    normalized_dfs = res[0]
    centroids_list = res[1]
    place_names = res[2]
    place_names_w_h = res[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clusters = []
    for i in range(len(normalized_dfs)):
        clusters.append(get_clusters(normalized_dfs[i], centroids_list[i]))

    run_all_cities(num_cities, place_names, place_names_w_h, normalized_dfs, centroids_list, clusters, output_dir)

    # run_only_states(num_states, normalized_dfs, centroids_list, clusters, place_names, output_dir)
    #
    # run_cities_same_state(num_cities, normalized_dfs, centroids_list, clusters, place_names, output_dir)
    #
    # run_cities_different_states(num_cities, normalized_dfs, centroids_list, clusters, place_names, output_dir)


if __name__ == '__main__':
    main()
