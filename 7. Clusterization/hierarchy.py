import numpy as np
import csv


class HierarchicCluster:
    def __init__(self, data):
        self.data = data

    def cluster(self):
        clusters = list()

        for record in self.data:
            clusters.append(record)

        for i in range(len(clusters)):
            mini = float('inf')
            a = 0
            b = 0

            for j in range(len(clusters)):
                for k in range(len(clusters)):
                    if j != k and self.distance(clusters[j], clusters[k]) < mini:
                        mini = self.distance(clusters[j], clusters[k])
                        a = j
                        b = k

            clusters[a] = clusters[a] + clusters[b]

            if b != 0 and b != len(clusters) - 1:
                clusters = clusters[0: b] + clusters[(b + 1): (len(clusters))]
            elif b == 0:
                clusters = clusters[1: (len(clusters))]
            else:
                clusters = clusters[0: (len(clusters) - 1)]

            if len(clusters) < 3:
                break

        return clusters

    @staticmethod
    def distance(a, b):
        dist = 0

        for x in a:
            for y in b:
                dist += np.linalg.norm(np.subtract(x, y))

        return dist / float(len(a) * len(b))


reader = csv.reader(open('train.csv'))
next(reader)
data = list()
features = list()

for row in reader:
    row[1] = float(row[1]) / 2.0

    if row[4] == 'male':
        row[4] = 0.0
    else:
        row[4] = 0.5

    row[9] = float(row[9]) / 300.0
    features.append(row[9])

    del row[11]
    del row[10]
    del row[8]
    del row[7]
    del row[6]
    del row[5]
    del row[3]
    del row[2]
    del row[0]

    row = np.array(row).astype('float32')

    data.append(row)

hierarchy = HierarchicCluster(data)
clusters = hierarchy.cluster()

print(clusters, file=open('clusters', 'w'))
