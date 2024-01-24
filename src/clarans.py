from functools import partial
from random import sample, choice, seed

from tqdm import tqdm

import metrics as dm
import numpy as np


class CLARANS:
    def __init__(self, data, k, numlocal, maxneighbour, distance_metric='euclidean', class_column='class'):
        seed(42)
        self.dist_metric = distance_metric

        self.data = data
        self.class_column = class_column
        self.indices = range(len(self.data))

        # 1. Zdefiniować parametry
        self.k = k
        self.numlocal = numlocal
        self.maxneighbour = maxneighbour

        self.i = 1
        self.mincost = np.inf

        self.bestnode = None
        self.current = None

        self.ranges = None
        if self.dist_metric == 'gower':
            self.ranges = self.get_ranges()

        self.metrics = {
            'euclidean': dm.d_euclidean,
            'manhattan': dm.d_manhattan,
            'minkowski': dm.d_minkowski,
            'cosine': dm.d_cosine,
            'hamming': dm.d_hamming,
            'jaccard': dm.d_jaccard,
            'mahalanobis': dm.d_mahalanobis,
            'gower': partial(dm.d_gower, ranges=self.ranges, omit_column=self.class_column)
        }

    def get_ranges(self):
        r = []
        for n in self.data.dtype.names:
            if n != self.class_column:
                if np.issubdtype(self.data[n].dtype, np.number):
                    r.append((n, (min(self.data[n]), max(self.data[n]))))
                else:
                    r.append((n, None))
        return r

    def distance(self, p1, p2):
        if self.dist_metric != 'gower':
            p1 = np.array([float(p1[n]) for n in p1.dtype.names[:-1] if n != self.class_column])
            p2 = np.array([float(p2[n]) for n in p2.dtype.names[:-1] if n != self.class_column])

        return self.metrics.get(self.dist_metric, self.metrics['euclidean'])(p1, p2)

    def generate_clusters(self, medoids):
        clusters = {i: [] for i in medoids}
        for i, p in enumerate(self.data):
            min_d = np.inf
            closest_medoid = None
            for m in medoids:
                if i != m:
                    d = self.distance(p, self.data[m])
                    if d < min_d:
                        min_d = d
                        closest_medoid = m
            clusters[closest_medoid].append(i)
        return clusters

    def calculate_cost(self, medoids):
        t = 0
        clusters = self.generate_clusters(medoids)
        for m in medoids:
            for i in clusters[m]:
                t += self.distance(self.data[m], self.data[i])
        return t

    def cluster(self):
        """Returns indices of medoids in data"""

        for _ in tqdm(range(self.numlocal), desc='num_local'):
            # 2. Wybrać dowolny wierzchołek jako current
            self.current = sample(self.indices, self.k)
            current_cost = self.calculate_cost(self.current)

            # 3. Ustalić licznik rozważonych sąsiadów
            for _ in tqdm(range(self.maxneighbour), desc='maxneighbour'):
                # 4. Wybrać dowolnego sąsiada różniącego się jednym punktem
                neighbour = self.current.copy()
                random_point = choice(self.indices)
                random_medoid = choice(self.current)
                neighbour[neighbour.index(random_medoid)] = random_point

                # 4. Obliczyć koszt sąsiada
                neighbour_cost = self.calculate_cost(neighbour)

                # 5. Jeśli koszt sąsiada mniejszy zapisać go jako current
                if neighbour_cost < current_cost:
                    self.current = neighbour
                    current_cost = neighbour_cost
                # 6. W przeciwnym wypadku rozpatrywać kolejnego sąsiada
                else:
                    pass

            # 7. Jeśli obecny koszt mniejszy, zapisać najlepsze medoidy
            if current_cost < self.mincost:
                self.bestnode = self.current
                self.mincost = current_cost

        # 8. Po znalezieniu maksymalnej liczby minimum lokalnych, zwrócić bestnode
        return self.bestnode


if __name__ == '__main__':
    data = np.array([1, 3, 10, 25, 11, 12, 2, 20, 30])
    k = 3
    num_local = 5
    max_neighbors = 10

    c = CLARANS(data, k, num_local, max_neighbors, )
    best_medoids = c.cluster()
    best_clusters = c.generate_clusters(best_medoids)
    print("Best Medoids:", best_medoids)
    print("Best Clusters:", best_clusters)
