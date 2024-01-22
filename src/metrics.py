import numpy as np


# TODO: check metrics on 1dim and ndim data

def d_euclidean(p1, p2):
    return np.nan_to_num(np.linalg.norm(p2 - p1), nan=0)


def d_manhattan(p1, p2):
    return np.sum(np.abs(p2 - p1))


def d_minkowski(p1, p2, order=2):
    return np.power(np.sum(np.power(np.abs(p2 - p1), order)), 1 / order)


def d_cosine(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def d_hamming(p1, p2):
    # TODO: add condition
    return np.sum(p1 != p2)


def d_jaccard(p1, p2):
    if isinstance(p1, set) and isinstance(p2, set):
        return len(p1.intersection(p2)) / len(p1.union(p2))


def d_mahalanobis(p1, p2):
    covariance_matrix = np.cov(np.vstack([p1, p2]), rowvar=False)
    return np.sqrt(np.dot(np.dot((p2 - p1).T, np.linalg.inv(covariance_matrix)), (p2 - p1)))


def d_gower(p1, p2, ranges, omit_column, weight=1):
    dissimilarity_score = 0
    total_weight = 0
    for n, r in ranges:
        if n != omit_column:
            if r:
                if p1[n] and p2[n]:
                    diff = np.abs(p1[n] - p2[n]) / (r[1] - r[0])
                else:
                    diff = 0
                    weight = 0
            else:
                diff = 0 if p1[n] == p2[n] else 1
            dissimilarity_score += diff * weight
            total_weight += weight
    d = np.sqrt(1 - dissimilarity_score/total_weight)
    return np.nan_to_num(d, nan=0)
