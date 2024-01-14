import numpy as np


def d_euclidean(p1, p2):
    return np.linalg.norm(p2 - p1)


def d_manhattan(p1, p2):
    return np.sum(np.abs(p2 - p1))


def d_minkowski(p1, p2, order=2):
    return np.power(np.sum(np.power(np.abs(p2 - p1), order)), 1 / order)


def d_cosine(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def d_hamming(p1, p2):
    #TODO: add condition
    return np.sum(p1 != p2)


def d_jaccard(p1, p2):
    if isinstance(p1, set) and isinstance(p2, set):
        return len(p1.intersection(p2)) / len(p1.union(p2))


def d_mahalanobis(p1, p2):
    covariance_matrix = np.cov(np.vstack([p1, p2]), rowvar=False)
    return np.sqrt(np.dot(np.dot((p2 - p1).T, np.linalg.inv(covariance_matrix)), (p2 - p1)))
