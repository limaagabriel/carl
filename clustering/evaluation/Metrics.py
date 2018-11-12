import numpy as np
from scipy.spatial import distance
from operator import truediv
from numpy import inf
from sklearn.metrics import silhouette_score
import math

class Metrics(object):

    @staticmethod
    def get_clusters(data, centroids):
        # Clusterizar a partir de centroides
        clusters = {}
        for i in centroids:
            clusters[i] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            idx = dist.index(min(dist))
            clusters[idx].append(xi)
        return clusters

    @staticmethod
    def clustering_evaluation(criteria, centroids=None, data=None, clf=None, m=None, number_neighbors=None, centroidUnique=None):
        if criteria == 'gap' or criteria == '1':
            return Metrics.gap_statistic(data=data, clf=clf, centroids=centroids)
        elif criteria == 'quantizationError' or criteria == '2':
            return Metrics.quantization_error(data=data, centroids=centroids)
        elif criteria == 'sumInterClusterDistance' or criteria == '3':
            return Metrics.inter_cluster_statistic(centroids)
        elif criteria == 'clusterSeparationCrisp' or criteria == '4':
            return Metrics.cluster_separation_crisp(data=data, centroids=centroids)
        elif criteria == 'fuzzyClusterSeparration' or criteria == '5':
            return Metrics.cluster_separation_fuzzy(data=data, centroids=centroids, m=m)
        elif criteria == 'averageBetweenGroupSumOfSquare' or criteria == '6':
            return Metrics.abgss(data=data, centroids=centroids)
        elif criteria == 'edgeIndex' or criteria == '7':
            return Metrics.edge_index(data=data, centroids=centroids, number_neighbors=number_neighbors)
        elif criteria == 'clusterConnectedness' or criteria == '8':
            return Metrics.cluster_connectedness(data=data, centroids=centroids, number_neighbors=number_neighbors)
        elif criteria == 'intraClusterStatistic' or criteria == '9':
            return Metrics.intra_cluster_statistic(data=data, centroids=centroids)
        elif criteria == 'ballHall' or criteria == '10':
            return Metrics.ball_hall(data=data, centroids=centroids)
        elif criteria == 'jIndex' or criteria == '11':
            return Metrics.j_index(data=data, centroids=centroids, m=m)
        elif criteria == 'totalWithinClusterVariance' or criteria == '12':
            return Metrics.total_within_cluster_variance(data=data, centroids=centroids)
        elif criteria == 'classificationEntropy' or criteria == '13':
            return Metrics.classification_entropy(data=data, centroids=centroids, m=m)
        elif criteria == 'intraClusterEntropy' or criteria == '14':
            return Metrics.intra_cluster_entropy(data=data, centroids=centroids)
        elif criteria == 'calinskiHarabaszIndex' or criteria == '15':
            return Metrics.variance_based_ch(data=data, centroids=centroids)
        elif criteria == 'hartiganIndex' or criteria == '16':
            return Metrics.hartigan_index(data=data, centroids=centroids)
        elif criteria == 'xuIndex' or criteria == '17':
            return Metrics.variance_based_ch(data=data, centroids=centroids)
        elif criteria == 'ratkowskyLanceIndex' or criteria == '18':
            return Metrics.rl(data=data, centroids=centroids)
        elif criteria == 'wbIndex' or criteria == '19':
            return Metrics.wb(data=data, centroids=centroids)
        elif criteria == 'xieBeni' or criteria == '20':
            return Metrics.xie_beni(data=data, centroids=centroids, m=m)
        elif criteria == 'iIndex' or criteria == '21':
            return Metrics.i_index(data=data, centroids=centroids, centroidUnique=centroidUnique)
        elif criteria == 'dunnIndex' or criteria == '22':
            return Metrics.dunn_index(data=data, centroids=centroids)
        elif criteria == 'daviesBouldin' or criteria == '23':
            return Metrics.davies_bouldin(data=data, centroids=centroids)
        elif criteria == 'csMeasure' or criteria == '24':
            return Metrics.cs_index(data=data, centroids=centroids)
        elif criteria == 'silhouette' or criteria == '25':
            return Metrics.silhouette(data=data, centroids=centroids)
        else:
            raise Exception("Ivalid metric selected: {}".format(criteria))

    @staticmethod
    def quantization_error(data, centroids):

        clusters = {}
        for k in centroids:
            clusters[k] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        global_intra_cluster_sum = 0.0

        number_of_effective_clusters = 0

        for c in range(len(centroids)):
            partial_intra_cluster_sum = 0.0
            normalized_partial_intra_cluster_sum = 0.0


            if len(clusters[c]) > 0:
                number_of_effective_clusters = number_of_effective_clusters + 1
                for point in clusters[c]:
                    partial_intra_cluster_sum += np.linalg.norm(point - centroids[c])

                normalized_partial_intra_cluster_sum = (partial_intra_cluster_sum / float(len(clusters[c])))

            global_intra_cluster_sum += normalized_partial_intra_cluster_sum

        if number_of_effective_clusters > 0:
            normalized_global_intra_cluster_sum = (global_intra_cluster_sum / float(number_of_effective_clusters))
            return normalized_global_intra_cluster_sum
        else:
            raise Exception("Arithmetic Error: Division By Error!")

    @staticmethod
    def compute_Wk(data, centroids, clusters):
        # Funcao do Gap Statistic
        wk_ = 0.0
        for r in range(len(clusters)):
            nr = len(clusters[r])
            if nr > 1:
                dr = distance.pdist(clusters[r], metric='sqeuclidean')
                dr = distance.squareform(dr)
                dr = sum(np.array(dr, dtype=np.float64).ravel())
                wk_ += (1.0 / (2.0 * nr)) * dr
        return wk_

    @staticmethod
    def gap_statistic(data, clf, centroids):
        clusters1 = Metrics.get_clusters(data, centroids)
        random_data = np.random.uniform(0, 1, data.shape)
        clf.fit(random_data)
        centroids_ = clf.centroids
        clusters2 = Metrics.get_clusters(data, centroids_)
        Wk = Metrics.compute_Wk(data, centroids, clusters1)
        En_Wk = Metrics.compute_Wk(random_data, centroids_, clusters2)
        return np.log(En_Wk) - np.log(Wk)

    @staticmethod
    def gap_statistic_2(data, clf, centroids):
        n_ref = 10
        clusters1 = Metrics.get_clusters(data, centroids)
        gap = 0.0
        sum_l = 0.0
        logwk = []
        Wk = Metrics.compute_Wk(data, centroids, clusters1)
        for ref in range(n_ref):
            random_data = np.random.uniform(0, 1, data.shape)
            clf.fit(random_data)
            centroids_ = clf.centroids
            clusters2 = Metrics.get_clusters(data, centroids_)
            En_Wk = Metrics.compute_Wk(random_data, centroids_, clusters2)
            logwk.append(np.log(Wk))
            gap += (np.log(En_Wk) - logwk[ref])
            sum_l += np.log(En_Wk)
        sum_l = truediv(sum_l, n_ref)
        for ref in range(n_ref):
            result = abs(logwk[ref] - sum_l)
        ans = (truediv(1, n_ref) * (result ** truediv(1, 2.0))) * ((1+truediv(1,n_ref))**2.0)
        # print ans
        return ans

    @staticmethod
    def intra_cluster_statistic(data, centroids):
        # minimization
        clusters = {}
        for k in centroids:
            clusters[k] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        inter_cluster_sum = 0.0
        for c in centroids:
            if len(clusters[c]) > 0:
                for point in clusters[c]:
                    inter_cluster_sum += np.linalg.norm(point - centroids[c])
        return inter_cluster_sum



    @staticmethod
    def inter_cluster_statistic(centroids):
        centers = []
        #cluster = len(centroids)
        for c in centroids:
            centers.append(centroids[c])
        centers = np.array(centers, dtype=float)
        centers = distance.pdist(centers, metric='sqeuclidean')
        centers = distance.squareform(centers)
        centers = sum(np.array(centers, dtype=np.float64).ravel())
        #centers = (1.0 / cluster) * (1.0 / (cluster - 1)) * centers
        return centers

    @staticmethod
    def ball_hall(data, centroids):
        # minimization
        n_k = len(centroids)
        return truediv(Metrics.intra_cluster_statistic(data, centroids), n_k)

    @staticmethod
    def pertinence(data, centroids, m):
        degree_of_membership = np.zeros((len(data), len(centroids)))
        for idx_ in centroids:
            for idx, xi in enumerate(data):
                updated_degree_of_membership = 0.0
                norm = np.linalg.norm(xi - centroids[idx_])
                all_norms = [norm / np.linalg.norm(xi - centroids[c]) for c in centroids]
                all_norms = np.power(all_norms, 2 / (m - 1))
                updated_degree_of_membership = 1 / sum(all_norms)
                degree_of_membership[idx][idx_] = updated_degree_of_membership
        return degree_of_membership

    @staticmethod
    def j_index(data, centroids, m):
        # maximization
        j = 0.0
        n_d = len(data)
        n_k = len(centroids)
        if m <= 0.0:
            m = 2.0
        pertinence = Metrics.pertinence(data, centroids, m)
        for k in range(n_k):
            for i in range(n_d):
                j += pertinence[i][k] * distance.euclidean(data[i], centroids[k])
        return j

    @staticmethod
    def classification_entropy(data, centroids, m):
        # minimization
        pertinence = Metrics.pertinence(data, centroids, m)
        ce = 0.0
        for k in range(len(centroids)):
            for d in range(len(data)):
                if pertinence[d][k] != 0.0:
                    ce += pertinence[d][k]*math.log(pertinence[d][k])
        return truediv(ce, len(data))

    @staticmethod
    def intra_cluster_entropy(data, centroids):
        h = 0.0
        for k in range(len(centroids)):
            h += (1 - Metrics.intra_cluster_entropy_h(data, centroids[k], k) * Metrics.intra_cluster_entropy_g(data, centroids[k], k)) ** truediv(1, len(centroids))
        return h

    @staticmethod
    def intra_cluster_entropy_h(data, centroid_k, k):
        g = Metrics.intra_cluster_entropy_g(data, centroid_k, k)
        return -((g * math.log(g, 2)) + ((1 - g) * (math.log(1 - g))))

    @staticmethod
    def intra_cluster_entropy_g(data, centroid_k, k):
        g = 0.0
        for i in range(len(data)):
            g += (0.5 + truediv(Metrics.co(data[i], centroid_k), 2))
        return truediv(g, len(data))

    @staticmethod
    def co(data_i, centroids_k):
        co = 0.0
        data_sum = 0.0
        centroids_sum = 0.0
        for d in range(len(centroids_k)):
            co += data_i[d] * centroids_k[d]
            data_sum += data_i[d] ** 2
            centroids_sum += centroids_k[d] ** 2
        data_sum **= truediv(1, 2)
        centroids_sum **= truediv(1, 2)
        if data_sum == 0.0 or centroids_sum == 0.0:
            return 0.0
        co = truediv(co, data_sum * centroids_sum)
        return co

    @staticmethod
    def total_within_cluster_variance(data, centroids):
        # minimization
        twcv = 0.0
        n_d = len(data)
        n_c = len(centroids)
        n_dim = len(centroids[0])
        data_sum = 0.0
        data_sum_by_cluster = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        for k in range(n_c):
            for d in range(n_dim):
                by_dimension = 0.0
                for i in range(len(clusters[k])):
                    by_dimension += clusters[k][i] ** 2
            if len(clusters[k]) == 0.0:
                data_sum_by_cluster = 0.0
            else:
                data_sum_by_cluster += truediv(by_dimension, len(clusters[k]))
        for i in range(n_d):
            for d in range(n_dim):
                data_sum += data[i][d] ** 2.0
        data_sum -= data_sum_by_cluster
        return data_sum

    @staticmethod
    def variance_based_ch(data, centroids):
        return truediv(len(data) - len(centroids), len(centroids) - 1) * truediv(Metrics.inter_cluster_statistic(centroids), Metrics.intra_cluster_statistic(data,centroids))

    @staticmethod
    def hartigan(data, centroids):
        return math.log(truediv(Metrics.inter_cluster_statistic(centroids), Metrics.intra_cluster_statistic(data, centroids)))

    @staticmethod
    def xu(data, centroids):
        return (len(centroids[0]) * math.log((truediv(Metrics.intra_cluster_statistic(data, centroids), len(centroids[0]) * (len(data) ** 2))) ** truediv(1, 2))) + math.log(len(centroids))

    @staticmethod
    def wb(data, centroids):
        return (len(centroids), truediv(Metrics.intra_cluster_statistic(data, centroids), Metrics.inter_cluster_statistic(centroids)))

    @staticmethod
    def rl(data, centroids):
        dimensions = len(centroids[0])
        rl = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        for d in range(dimensions):
            for k in range(0,len(centroids)):
                for l in range(1,len(centroids)):
                    if k != l:
                        rl += truediv(Metrics.sum_c_c(centroids, k, l), Metrics.sum_x_c(clusters, centroids, k)) ** truediv(1, 2.0)
        return truediv(rl, len(centroids)**truediv(1,2.0))

    @staticmethod
    def sum_c_c(centroidsx, k, l):
        s = 0.0
        for d in range(len(centroidsx[k])):
            s += (centroidsx[k][d] - centroidsx[l][d]) ** 2.0
        return s ** truediv(1,2.0)

    @staticmethod
    def sum_x_c(clusters, centroids, k):
        s = 0.0
        for d_1 in range(len(clusters[k])):
            for d_2 in range(len(centroids[k])):
                s += (clusters[k][d_1] - centroids[k][d_2]) ** 2.0
        return s ** truediv(1, 2.0)

    @staticmethod
    def davies_bouldin(data, centroids):
        n_c = len(centroids)
        n_d = len(data)
        db = 0.0
        for k in range(n_c):
            db += Metrics.maximum_d_b(data, centroids, k)
        return truediv(db, n_c)

    @staticmethod
    def maximum_d_b(data, centroids, k):
        max_distance = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        for l in range(len(centroids)):
            if k != l:
                db = truediv(Metrics.sc_r(centroids, clusters, k) + Metrics.sc_r(centroids, clusters, l), distance.euclidean(centroids[k], centroids[l]) + 0.000000000000000000001)
                if max_distance < db:
                    max_distance = db
        return max_distance

    @staticmethod
    def sc_r(centroids, clusters, k):
        sc = 0.0
        for i in range(len(clusters[k])):
            sc += distance.euclidean(clusters[k][i], centroids[k])
        if len(clusters[k]) == 0:
            return 0.0
        return truediv(sc, len(clusters[k]))

    @staticmethod
    def xie_beni(data, centroids, m):
        pertinence = Metrics.pertinence(data, centroids, m)
        xb = 0.0
        for k in range(len(centroids)):
            for i in range(len(data)):
                xb += pertinence[i][k]*(distance.euclidean(data[i], centroids[k])**2)
        xb = truediv(xb, len(data)*Metrics.minimum_distance_centroids_power_2(centroids))
        return xb

    @staticmethod
    def minimum_distance_centroids_power_2(centroids):
        minx = distance.euclidean(centroids[0],centroids[1])**2
        for l in range(1, len(centroids)-1):
            for r in range(l+1, len(centroids)):
                if l != r:
                    md = distance.euclidean(centroids[l],centroids[r])**2
                    if minx > md:
                        minx = md
        return minx

    @staticmethod
    def maximum_distance_centroids(centroids):
        max = distance.euclidean(centroids[0],centroids[1])**2
        for l in range(1, len(centroids)-1):
            for r in range(l+1, len(centroids)):
                md = distance.euclidean(centroids[l],centroids[r])**2
                if(max < md):
                    max = md
        return max

    @staticmethod
    def i_index(data, centroids, centroidUnique):
        return (truediv(1, len(data))*truediv(Metrics.e1(data, centroidUnique), Metrics.j_index(data, centroids, 2.0))*Metrics.maximum_distance_centroids(centroids)) ** 2.0

    @staticmethod
    def e1(data, centroid):
        e1 = 0.0
        for d in range(len(data)):
            e1 += distance.euclidean(data[d], centroid)
        return e1

    @staticmethod
    def dunn_index(data, centroids):
        di = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        min_l = float('inf')
        min_k = float('inf')
        max_k = 0.0
        for k in range(len(centroids)):
            x = Metrics.maximum_distance_between_data_between_centroids(clusters[k])
            if max_k < x:
                max_k = x
        for k in range(0, len(centroids)-1):
            for l in range(1, len(centroids)):
                if k != l:
                    md = truediv(Metrics.minimum_distance_between_data_between_centroids(clusters[k], clusters[l]), max_k)
                    if min_l > md:
                        min_l = md
            if min_k > min_l:
                min_k = min_l
        return min_k

    @staticmethod
    def minimum_distance_between_data_between_centroids(cluster_1, cluster_2):
        min_de = float('inf')
        for data_centroid_one in range(len(cluster_1)):
            for data_centroid_two in range(len(cluster_2)):
                de = distance.euclidean(cluster_1[data_centroid_one], cluster_2[data_centroid_two])
                if min_de > de:
                    min_de = de
        return min_de

    @staticmethod
    def maximum_distance_between_data_between_centroids(cluster_1):
        max = 0.0
        for data_centroid_one in range(len(cluster_1)):
            for data_centroid_two in range(len(cluster_1)):
                de = distance.euclidean(cluster_1[data_centroid_one], cluster_1[data_centroid_two])
                if max < de:
                    max = de
        return max


    @staticmethod
    def silhouette(data, centroids):
        sil = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        for k in range(len(clusters)):
            for d_out in range(len(clusters[k])):
                ai = Metrics.silhouette_a(clusters[k][d_out], k, clusters)
                bi = Metrics.minimum_data_data(clusters[k][d_out], k, clusters)
                max_a_b = max(ai, bi)

                sil += truediv(bi - ai, max_a_b + 0.00001)
        return truediv(sil, len(data))


    @staticmethod
    def silhouette_a(datum, cluster_in, clusters):
        sum_d = 0.0
        for d_out in range(len(clusters[cluster_in])):
            sum_d += distance.euclidean(datum, clusters[cluster_in][d_out])
        sum_d = truediv(sum_d, len(clusters[cluster_in]) + 0.000000000000000000001)
        return sum_d

    @staticmethod
    def minimum_data_data(datum, cluster_in, clusters):
        min_D = float('inf')
        for k in range(len(clusters)):
            if cluster_in != k:
                x = 0.0
                for d_out in range(len(clusters[k])):
                        x += distance.euclidean(datum, clusters[k][d_out])
                x = truediv(x, len(clusters[k])+0.000000000000000000001)
                if min_D > x:
                    min_D = x
        return min_D

    @staticmethod
    def cs_index(data, centroids):
        cs = 0.0
        min_sum = 0.0
        max_k = 0.0
        min_k = distance.euclidean(centroids[0], centroids[1])
        clusters = Metrics.get_clusters(data, centroids)
        for k in range(len(centroids)):
            x = Metrics.maximum_distance_between_data_between_centroids(clusters[k])
            if max_k < x:
                max_k = x
            for l in range(len(centroids)):
                if k != l:
                    y = distance.euclidean(centroids[k], centroids[l])
                    if min_k > y:
                        min_k = y
            min_sum += min_k
        for k in range(len(centroids)):
            cs += truediv(max_k, len(clusters[k]))
        cs = truediv(cs, min_sum)
        return cs

    @staticmethod
    def min_max_cut(data, centroids):
        mmc = 0.0
        mmc_den = 0.0
        clusters = Metrics.get_clusters(data, centroids)
        for k1 in range(len(clusters)):
            for k2 in range(len(clusters)):
                for kk1 in range(len(clusters[k1])):
                    for kk2 in range(len(clusters[k2])):
                        if k1 != k2:
                            mmc += distance.euclidean(clusters[k1][kk1], clusters[k2][kk2])
                        if k1 == k2:
                            mmc_den += distance.euclidean(clusters[k1][kk1], clusters[k2][kk2])
        mmc = truediv(mmc, mmc_den)
        return -mmc

    @staticmethod
    def cluster_connectedness(data, centroids, number_neighbors):
        cc = 0.0
        np.sort(data, axis=0)
        clusters = Metrics.get_clusters(data, centroids)
        for d in range(len(data)-number_neighbors):
            index_1 = idx = Metrics.ismember(data[d], clusters)
            for neighbor in range(1, number_neighbors):
                index_2 = idx = Metrics.ismember(data[d+neighbor], clusters)
                if index_1 != index_2:
                    cc += truediv(1, neighbor)
        return cc

    @staticmethod
    def cluster_separation_fuzzy(data, centroids, m):
        cs = 0.0
        n_k = len(centroids)
        if m <= 0.0:
            m = 2.0
        pertinence = Metrics.pertinence_cluster_separation(centroids, m)
        for k in range(n_k):
            for l in range(n_k):
                if k != l:
                    cs += pertinence[k][l] * distance.euclidean(centroids[l], centroids[k])
        return cs

    @staticmethod
    def pertinence_cluster_separation(centroids, m):
        degree_of_membership = np.zeros((len(centroids), len(centroids)))
        for idx_ in centroids:
            for idx in centroids:
                if idx_ != idx:
                    norm = np.linalg.norm(centroids[idx_] - centroids[idx])
                    all_norms = np.array([norm /np.linalg.norm(centroids[idx_] - centroids[c]) for c in centroids])
                    while np.isposinf(all_norms).any() or np.isneginf(all_norms).any():
                        all_norms[all_norms == inf] = 0
                        all_norms[all_norms == -inf] = 0
                    all_norms = np.power(all_norms, truediv(2.0, (m - 1.0)))
                    degree_of_membership[idx][idx_] = truediv(1.0, sum(all_norms))
                    #print degree_of_membership[idx][idx_]
        return degree_of_membership

    @staticmethod
    def cluster_separation_crisp(data, centroids):
        x = len(centroids)*(len(centroids)-1)
        return truediv(2.0, x) * Metrics.inter_cluster_statistic(centroids)

    @staticmethod
    def abgss(data, centroids):
        clusters = Metrics.get_clusters(data, centroids)
        abgss = 0.0
        for k in range(len(centroids)):
            euclidean = 0.0
            for d in range(len(centroids[k])):
                euclidean += (centroids[k][d] - np.mean(data[d])) ** 2.0
            abgss += len(clusters[k])*euclidean
        return truediv(abgss, len(centroids))

    @staticmethod
    def edge_index(data, centroids, number_neighbors):
        ei = 0.0
        np.sort(data, axis=0)
        clusters = Metrics.get_clusters(data, centroids)
        for d in range(len(data)-number_neighbors):
            idx = Metrics.ismember(data[d], clusters)
            for neighbor in range(1, number_neighbors):
                idx2 = Metrics.ismember(data[d+neighbor], clusters)
                if idx != idx2:
                    ei += distance.euclidean(data[d], data[d+neighbor])
        return -ei

    @staticmethod
    def ismember(d, c):
        for a1 in range(len(c)):
            for a2 in range(len(c[a1])):
                if np.array_equal(c[a1][a2], d):
                    return a1

    @staticmethod
    def hartigan_index(data, centroids):
        return np.log(Metrics.inter_cluster_statistic(centroids) / Metrics.intra_cluster_statistic(data, centroids))
