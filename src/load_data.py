import numpy as np
import pandas as pd
import scipy.io as io
import glob
import networkx as nx
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA

from src.clarans import CLARANS

pio.renderers.default = 'png'
pd.options.plotting.backend = 'plotly'


def read_file(path):
    return io.arff.loadarff(path)


# def create_complete_graph(data_pts):
#     return nx.complete_graph(data_pts)
def cluster(data, k=3, p=0.015, num_local=2):
    # TODO: ensure data is array?
    max_neighbors = int(p*k*(len(data)-k))
    # if len(set([data[n].dtype for n in data.dtype.names])) > 1:  # different data types
    #     distance_metric = 'gower'
    # else:
    #     distance_metric = 'euclidean'
    distance_metric = 'euclidean'
    c = CLARANS(data, k, num_local, max_neighbors, distance_metric)
    best_medoids = c.cluster()
    best_clusters = c.generate_clusters(best_medoids)
    return best_clusters


def plot_clustering(df, color_column='cluster', chart_name='chart.png'):
    fig = (df
           # .sort_values(by=color_column)
           .plot(kind='scatter', x='x', y='y', color=color_column, color_discrete_sequence=px.colors.qualitative.Set1))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.write_image(chart_name)


def evaluate(df, best_clusters, chart_name='chart.png'):
    medoids = best_clusters.keys()
    # class_labels = df.loc[medoids, 'class'].to_dict()
    values = [(val, key) for key, values in best_clusters.items() for val in values]
    cluster_df = pd.DataFrame(values, columns=['index', 'value'])
    cluster_df.set_index('index', inplace=True)
    # cluster_df = cluster_df.replace(class_labels)
    cluster_df['value'] = cluster_df['value'].astype('str')
    df['cluster'] = cluster_df['value']
    df.to_csv(chart_name.replace('.png', '.csv'))
    if 'x' in df.columns and 'y' in df.columns:
        plot_clustering(df, 'cluster', chart_name.replace('.png', '_clarans.png'))
        plot_clustering(df, 'class', chart_name.replace('.png', '_default.png'))

    # plt.scatter(df['x'],df['y'], c=df['cluster'], cmap='Accent')
    # plt.show()


if __name__ == '__main__':
    files = glob.glob("../data/*.arff")
    print(files)
    # for f in files:
    #     data, meta = read_file(f)
    #     # print(data)
    #     print(meta)
    f = files[2]
    data, meta = read_file(f)
    print(meta)
    k = 6
    p = 0.015
    best_clusters = cluster(data, k=k, p=p)
    #
    df = pd.DataFrame(data, columns=meta.names())
    evaluate(df, best_clusters, chart_name=f'{f}_clustering_{k}_p_{p}.png')

