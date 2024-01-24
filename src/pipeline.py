import csv
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import scipy.io as io
import glob
import plotly.io as pio
import plotly.express as px
from sklearn.metrics import davies_bouldin_score, silhouette_score, rand_score
from sklearn.metrics.cluster import contingency_matrix

from src.clarans import CLARANS

pio.renderers.default = 'png'
pd.options.plotting.backend = 'plotly'


def read_file(path):
    return io.arff.loadarff(path)


def cluster(data, k=3, p=0.015, num_local=2, distance='euclidean'):
    max_neighbors = int(p * k * (len(data) - k))
    c = CLARANS(data, k, num_local, max_neighbors, distance)
    best_medoids = c.cluster()
    best_clusters = c.generate_clusters(best_medoids)
    return best_clusters


def plot_clustering(df, color_column='cluster', chart_name='chart.png'):
    if df['class'].nunique() >= 8:
        palette = px.colors.qualitative.Set1
    else:
        palette = px.colors.qualitative.Light24
    fig = (df
           # .sort_values(by=color_column)
           .plot(kind='scatter', x='x', y='y', color=color_column, color_discrete_sequence=palette))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.write_image(chart_name)


def evaluate(df, name, k, p, num_local, distance, t):
    fname = f'../results/{name}_{k}_{p}_{num_local}_{distance}'
    df.to_csv(f'{fname}.csv')

    X = df.drop(columns=['cluster', 'class']).fillna(0).values
    labels_true = df['class'].astype(int).values
    labels_predicted = df['cluster'].astype(int).values

    if distance == 'euclidean':
        db_score = davies_bouldin_score(X, labels_predicted)
    else:
        db_score = None

    s_score = silhouette_score(X, labels_predicted, metric=distance)

    c = contingency_matrix(labels_true, labels_predicted)
    purity_score = np.sum(np.amax(c, axis=0)) / np.sum(c)

    rnd_score = rand_score(labels_true, labels_predicted)

    row = [fname, db_score, s_score, purity_score, rnd_score, t]

    with open('../results/results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    if 'x' in df.columns and 'y' in df.columns:
        plot_clustering(df, 'cluster', f'{fname}_clarans.png')
        plot_clustering(df, 'class', f'{fname}_default.png')


def get_clustering_df(data, best_clusters, columns):
    df = pd.DataFrame(data, columns=columns)

    values = [(val, key) for key, values in best_clusters.items() for val in values]

    cluster_df = pd.DataFrame(values, columns=['index', 'value'])
    cluster_df.set_index('index', inplace=True)
    cluster_df['value'] = cluster_df['value'].astype('str')

    df['cluster'] = cluster_df['value']
    return df







