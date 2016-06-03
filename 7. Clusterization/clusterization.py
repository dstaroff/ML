from random import choice

import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.cluster_center = None

    def cluster(self, x):
        if self.cluster_center is None:
            self.cluster_center = np.array([choice(x)[0] for _ in range(self.k)])

        clusters = [list() for _ in range(self.k)]

        for i in range(self.k):
            for j in range(len(x)):
                lengths = np.array(
                    [np.linalg.norm(np.subtract(x[j][0], self.cluster_center[k])) for k in range(self.k)])

                mini = np.argmin(lengths)
                clusters[mini].append(x[j])

            if i != self.k - 1:
                for j in range(self.k):
                    summ = np.array([0 for _ in range(784)])

                    for h in clusters[j]:
                        summ = np.add(summ, h[0])

                    self.cluster_center[j] = np.divide(summ, len(clusters[j]))

                clusters = [[] for _ in range(self.k)]

        return clusters

    def set_clusters(self, clusters):
        self.cluster_center = clusters


class DBScan:
    def __init__(self, eps, n):
        self.eps = eps
        self.n = n
        self.clusters = list()

    def get_neighbours(self, record, data):
        neighbours = list()

        for another_record in data:
            if np.linalg.norm(np.subtract(record[0], another_record[0])) < self.eps:
                neighbours.append(another_record)

        return neighbours

    def expand_cluster(self, record, neighbours, cluster, data):
        cluster.append((record[0], record[1]))
        record[3] = 1

        for new_record in neighbours:
            if new_record[2] != 1 and new_record[2] != -1:
                new_record[2] = 1
                new_neighbors = self.get_neighbours(new_record, data)

                if len(new_neighbors) >= self.n:
                    neighbours = neighbours + new_neighbors

            if new_record[3] == 0:
                cluster.append((record[0], record[1]))
                new_record[3] = 1

    def scan(self, data):
        for i in range(len(data)):
            data[i] = [data[i][0], data[i][1], 0, 0]

        for record in data:
            if record[2] != 1 and record[2] != -1:
                record[2] = 1
                neighbours = self.get_neighbours(record, data)

                if len(neighbours) < self.n:
                    record[2] = -1
                else:
                    self.clusters.append(list())
                    self.expand_cluster(record, neighbours, self.clusters[len(self.clusters) - 1], data)

        return self.clusters


def cluster_response(cluster, k):
    c = [0.0 for _ in range(k)]

    for element in cluster:
        c[element[1]] += 1

    return np.argmax(np.asarray(c))


def score_k_means(data):
    means = KMeans(10)
    clusters = means.cluster(data)

    mistake = 0
    for cluster in clusters:
        local_mistake = 0
        response = cluster_response(cluster, 10)

        for element in cluster:
            if element[1] == response:
                local_mistake += 1

        mistake += local_mistake / float(len(cluster))

    return mistake / float(len(clusters))


def score_db_scan(data, eps, n):
    db_scan = DBScan(eps, n)
    clusters = db_scan.scan(data)

    mistake = 0
    for cluster in clusters:
        local_mistake = 0
        response = cluster_response(cluster, len(clusters))

        for element in cluster:
            if element[1] == response:
                local_mistake += 1

        mistake += local_mistake / float(len(cluster))

    return mistake / float(len(clusters))
