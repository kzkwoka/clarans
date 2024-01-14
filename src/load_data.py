import scipy.io as io
import glob
import networkx as nx
import matplotlib.pyplot as plt


def read_file(path):
    return io.arff.loadarff(path)


def create_complete_graph(data_pts):
    return nx.complete_graph(data_pts)


if __name__ == '__main__':
    files = glob.glob("../data/*.arff")
    # for f in files:
    #     data, meta = read_file(f)
    #     print(data)
    #     # print(meta)
    data, meta = read_file(files[0])
    # G = create_complete_graph(data)
    # print(type(data[0]))
    # dir(data)

    G = nx.Graph()
    for i, (x, y, c) in enumerate(data[:4]):
        G.add_node(i, x=x, y=y, cl=c)
    # Visualize the original graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=6, font_color='black',
            font_weight='bold', edge_color='gray', linewidths=0.7, alpha=0.7)
    plt.title("Not Connected Graph")
    plt.show()

    complete_G = nx.complete_graph(G.nodes())
    # Visualize the complete graph
    pos = nx.spring_layout(complete_G)
    nx.draw(complete_G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=6, font_color='black',
            font_weight='bold', edge_color='gray', linewidths=0.7, alpha=0.7)
    plt.title("Complete Graph")
    plt.show()

    line_graph = nx.line_graph(complete_G)

    # Visualize the line graph
    pos_line = nx.spring_layout(line_graph)
    nx.draw(line_graph, pos_line, with_labels=True, font_size=10, font_color='black', font_weight='bold',
            edge_color='gray', linewidths=1, alpha=0.7)
    plt.title("Line Graph (Edges become Nodes)")
    plt.show()
