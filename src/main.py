import glob
import time

from src.pipeline import read_file, cluster, get_clustering_df, evaluate


def assign_k(name):
    if name == 'ds4c2sc8':
        k = 8
    elif name == 'D31':
        k = 31
    elif name == 'dermatology':
        k = 6
    else:
        k = 3
    return k


if __name__ == '__main__':
    files = glob.glob("../data/*.arff")
    p_range = [0.25, 0.5, 1, 1.25, 1.5, 5]

    n_local_range = [1, 2, 3]
    metrics = ['euclidean', 'manhattan', 'cosine']
    files = [files[1]]
    for f in files:
        data, meta = read_file(f)
        k = assign_k(meta.name)
        print(meta.name, k)
        for p in p_range:
            if meta.name == 'D31' and p >= 1:
                continue
            p = p / 100
            for num_local in n_local_range:
                if meta.name == 'D31' and num_local != 2:
                    continue
                for metric in metrics:
                    if meta.name == 'D31' and metric != 'euclidean':
                        continue
                    start = time.time()
                    best_clusters = cluster(data, k=k, p=p, num_local=num_local, distance=metric)
                    end = time.time()
                    df = get_clustering_df(data, best_clusters, list(meta.names()))
                    evaluate(df, name=meta.name, k=k, p=p, num_local=num_local, distance=metric, t=end - start)
        # break
