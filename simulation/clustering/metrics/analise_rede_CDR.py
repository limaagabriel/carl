# coding=utf-8
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from operator import truediv
import scipy.sparse



def get_statistics(graph, matrix):
    plt.figure()
    d = matrix.sum(axis=1)
    e = np.sum(d) / graph.order()
    print("Average degree: ", e)

    hist = nx.degree_histogram(graph)
    entropy = 0
    for i in range(len(hist)):
        if hist[i] != 0:
            entropy -= hist[i] * np.log2(hist[i])

    print("Entropy: ", entropy)

    # average_sp = nx.average_shortest_path_length(graph)
    # print("Average shortest path length: ", average_sp)

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
    # plt.show()



import csv
import pandas as pd
os.chdir('../../../..')
Gm = nx.read_pajek("results/all_cities/graph.pajek")
file_name_0 = open("results/all_cities/label.csv", 'rb')
file_name_1 = open("results/all_cities/infomap.csv", 'rb')
file_name_2 = open("results/all_cities/louvain.csv", 'rb')
file_name_3 = open("results/all_cities/stochastic.csv", 'rb')
ids = []

for i in range(2):
    ids.append(file_name_0.readline().strip().split(','))

i = 0
G = nx.Graph(Gm)
for e in G.nodes():
    for n in range(0, 2):
        if n != 1:
            for df_i in ids[n]:
                if float(e.strip()) == float(df_i.strip()):
                    i = i + 1
                    G.remove_node(e)


A = nx.adjacency_matrix(G)
get_statistics(G, A)

# print "transividade"
# print nx.transitivity(G)
# print "clustering"
# print nx.clustering(G)
# print nx.load_centrality(G)
# nx.draw_networkx(G)
# plt.axis("tight")
# plt.show()
# get_statistics(G, A)
# BC = nx.betweenness_centrality(G)
# print list(nx.enumerate_all_cliques(G))
# print
# c = list(nx.find_cliques(G))
# print nx.graph_clique_number(G)
# print "node_clique"
# print len(nx.node_clique_number(G))
# print
# print len(c[0])
# print len(c[1])
# print len(c[2])
# print len(c[3])
# print len(c[4])

# print list(nx.all_node_cuts(G))
# print nx.average_node_connectivity(G)
# print nx.conductance(G)
# total_number_nodes = G.order()
# number_nodes = G.number_of_nodes()
# total_number_edges = G.size()
#
# print(str(total_number_nodes)+": Number of nodes")
# print(str(total_number_edges)+": Number of edges")
# print(str(G.degree())+" => Degrees")
#
#
# print('Calcular o grau médio do grafo')
# vector_degree = A.sum(axis=1)
# print(vector_degree)
# e = truediv(np.sum(vector_degree),len(vector_degree))
# print('Grau médio do grafo: '+str(e))
#
# d = np.array(A.sum(axis=0))
# D = np.diag(d[0])
# print("Matriz de Graus: \n")
# print(D)
#
# print('Distribuição dos graus dos nós: '+str(nx.degree_histogram(G)))
# # distribuicao = nx.degree_histogram(G)
# # plt.bar(range(len(distribuicao)),distribuicao)
# # plt.margins(0.05, 0.05)
# # plt.title(u"Distribuição dos graus")
# # plt.xlabel(u'Grau')
# # plt.ylabel(u'Número de graus')
# # plt.show()
#
# print('Densidade')
# print(nx.density(G))
#
# f = nx.info(G)
# print(f)
# # f = nx.info(G,2)
# # print(f)
#
#
# print('Matriz laplaciana')
# L= D-A
# print(L)
# matrix_laplaciana = nx.laplacian_matrix(G)
# print
# print(matrix_laplaciana)
# print
# autovalores = nx.laplacian_spectrum(G)
# print(autovalores)
# print('A quantidade de zeros é igual ao numero de componentes independentes do grafo (desconectados+1)')
# #
# # plt.bar(range(len(autovalores)),autovalores)
# # plt.title(u"Gráfico de autovalores")
# # plt.xlabel(u"Autovalores")
# # plt.show()
#
# print("Entropia da rede, aleatoriedade, grau de desordem")
# distribuicaoDosGraus = nx.degree_histogram(G)
# entropy = 0
# for i in range(0,len(distribuicaoDosGraus)):
#     if distribuicaoDosGraus[i] != 0:
#         entropy = entropy - distribuicaoDosGraus[i]*np.log2(distribuicaoDosGraus[i])
# print(entropy)
# print
#
# print("Matriz de menores caminhos: simetrico pq o grafo é não-direcionado")
# numeroNos = G.order()
# SP = nx.all_pairs_shortest_path_length(G)
# SPM = np.zeros((numeroNos,numeroNos))
# for i in range(0,numeroNos):
#     for j in range(0,numeroNos):
#         SPM[i][j] = SP[i][j]
# print(SPM)
#
# print
# print("\n menor caminho médio: somar todos os menores caminhos dividido pelo numero de menores caminhos")
# ASP = nx.average_shortest_path_length(G)
# print(ASP)
#
# print("\n Clustering de cada nó individualmente: mede o agrupamento dos nós pelas triangulações, se vc tem um nó com 3 ligacoes entao eles tem 3 ligacoes.. tem potencial de formar tres triangulos  mas nem sempre esta conectado ")
# print(" quantidade de triangulos possives C n,2 = n(n-1)/2 ")
# print(" clustering entao é igual a 2*(arestas)/vizinhos(vizinhos-1) ")
# C = nx.clustering(G)
# print(C)
# print("soma todos clustering e divide pelo numero de nós")
# AC = nx.average_clustering(G)
# print(AC)
#
# print("disortativo (hubs) / assortativo(grau de nó semelhante, rede altamente regular é assortativa)/ =0 aleatorio")
# Assortatividade = nx.degree_assortativity_coefficient(G)
# print(Assortatividade)
#
# print
# print("centralidade, maior quando for uma bridge")
# print("numero de caminhos que o nó i aparece dividido pelo numero total de caminhos existentes")
# BC = nx.betweenness_centrality(G)
# print(BC)
#
#
