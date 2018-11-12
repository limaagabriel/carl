__author__ = 'mgmmacedo'
import os
import networkx as nx

os.chdir('../../../..')
Gm = nx.read_pajek("results/all_cities/graph.pajek")
G = nx.Graph(Gm)

# print
# print "K-components"
# print "Returns the approximate k-component structure of a graph G."
# kc = nx.k_components(G)
# print kc

# print
# print "Assortativity"
# print "Compute degree assortativity of graph."
# dac = nx.degree_assortativity_coefficient(G)
# print dac
# print "Compute assortativity for numerical node attributes."
# nac = nx.numeric_assortativity_coefficient(G,'size')
# print nac
#
# print
# print "Transitivity"
# print "Compute graph transitivity, the fraction of all possible triangles present in G."
# dbc = nx.transitivity(G)
# print dbc

# print
# print "Clique"
# print "Find the Maximum Clique"
# mc = nx.max_clique(G)
# print mc

# print
# print "Average clustering"
# print "Estimates the average clustering coefficient of G."
# ac = nx.average_clustering(G)
# print ac

# print
# print "Maximum independent set"
# print "Return an approximate maximum independent set."
# mis = nx.maximum_independent_set(G)
# print mis
#
#
#
# print
# print "Average degree connectivity"
# print "Compute the average degree connectivity of graph."
# adc = nx.average_degree_connectivity(G)
# print adc

# print
# print "KNN"
# print "Compute the average degree connectivity of graph."
# knn = nx.k_nearest_neighbors(G)
# print knn


# print
# print "Density"
# print "Return density of bipartite graph B."
# d = nx.density(G)
# print d

# print
# print "Clustering"
# print "Compute a bipartite clustering coefficient for nodes."
# c = nx.clustering(G)
# print c

# print
# print "Average clustering"
# print "Compute the average bipartite clustering coefficient."
# ac = nx.average_clustering(G)
# print ac

# print
# print "Degree centrality"
# print "Compute the degree centrality for nodes in a bipartite network."
# dc = nx.degree_centrality(G)
# print dc

# print
# print "Out degree centrality"
# print "Compute the out-degree centrality for nodes"
# odc = nx.out_degree_centrality(G)
# print odc
#
# print
# print "In degree centrality"
# print "Compute the in-degree centrality for nodes."
# idc = nx.in_degree_centrality(G)
# print idc

# print
# print "Bridges"
# print "Decide whether a graph has any bridges."
# dbc = nx.has_bridges(G)
# print dbc

# print
# print "Clique"
# print "Returns all cliques in an undirected graph."
# dbc = nx.enumerate_all_cliques(G)
# print dbc
# print "Returns all maximal cliques in an undirected graph"
# dbc = nx.find_cliques(G)
# print dbc
# print "Returns the maximal clique graph of the given graph."
# dbc = nx.make_max_clique_graph(G)
# print dbc
# print "number of cliques"
# nbc = nx.graph_clique_number(G)
# print nbc
# nbc = nx.graph_number_of_cliques(G)
# print nbc

# print
# print "Triangles"
# print "Compute the number of triangles."
# dbc = nx.triangles(G)
# print len(dbc)



# print
# print "Node Connectivity"
# print "Returns an approximation for node connectivity for a graph."
# nc = nx.node_connectivity(G)
# print nc
#
# print
# print "Connectivity"
# print "Compute node connectivity between all pairs of nodes."
# apnc = nx.all_pairs_node_connectivity(G)
# print apnc