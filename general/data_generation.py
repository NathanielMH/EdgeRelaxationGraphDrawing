# Notebook to generate data and save it in filename specified by input argument.
from src.graph_utils import max_j_node_centrality, sum_j_node_centrality, j_node_centrality, graph_entropy_norm, electro_forces_in_neighbourhood, cos_force_diff_in_neighbourhood
from src.graph_utils import gradient_kamada_kawai, max_neighbour_degrees_norm, sum_neighbour_degrees_norm, expansion_factor_norm, edge_crossings_norm
from src.graph_utils import stress, total_stress, num_crossings, mean_edge_length, nodes_dict_to_array, distance_matrix
from src.graph_parser import parseGraphmlFile
from src.graph_dataset import GraphDataset
from fa.forceatlas2 import ForceAtlas2 as fa2
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# Generate Dataframe
n = 0
m = 2
benchmarks = ['random-dag', 'rome', 'north']


def read_list_of_graphs(dir_name, ext):
    list_graphs = [parseGraphmlFile(dir_name+f, weighted=False, directed=False)
                   for f in os.listdir(dir_name) if f.endswith('.' + ext)]
    return list_graphs


def draw_fa2(g): return fa2.forceatlas2_networkx_layout(
    g, pos=nx.spectral_layout(g), iterations=100)


def draw_kk(g): return nx.kamada_kawai_layout(g)


algo_dict = {'fa2': draw_fa2, 'kk': draw_kk}


def generate_data_from_list(list_graphs: list, bench: str, list_features: list, algo_name: str):
    """
    Generates a list of rows for a dataframe from the graphs in list_graphs.

    Args:
        list_graphs (list): list of graphs.
        bench (str): name of the benchmark.

    Returns:
        data (list): list of rows for df from list_graphs
    """
    data = []
    for idx_graph, graph in tqdm(list(enumerate(list_graphs[n:m]))):

        #  Run Spectral +  algorithm chosen
        pos0 = algo_dict[algo_name](graph, pos=nx.spectral_layout(graph))

        #  Compute general graph attributes
        eb = nx.edge_betweenness(graph)     # edge betweenness
        st = stress(graph, pos0)             # stress
        cross0 = num_crossings(graph, pos0)
        edgel0 = mean_edge_length(graph, pos0)
        total_stress0 = total_stress(graph, pos0)
        deg = nx.degree(graph, graph.nodes)
        bridges = nx.bridges(graph)
        d0 = distance_matrix(graph)
        graph_entropy = graph_entropy_norm(graph)
        pos0_arr = nodes_dict_to_array(pos0)
        r = 0.1

        for idx_edge, e in enumerate(graph.edges):

            n1, n2 = e

            # force in neighbourhood of n1 before relaxation
            force_before_1 = electro_forces_in_neighbourhood(
                graph, n1, pos0, r)
            # force in neighbourhood of n2 before relaxation
            force_before_2 = electro_forces_in_neighbourhood(
                graph, n2, pos0, r)
            force_before = force_before_1 + force_before_2
            # New position removing edge
            graph_copy = graph.copy()
            graph_copy.remove_edges_from([e])
            pos1 = algo_dict[algo_name](graph_copy, pos=pos0)
            pos1_arr = nodes_dict_to_array(pos1)

            cross1 = num_crossings(graph, pos1)
            edgel1 = mean_edge_length(graph, pos1)
            total_stress1 = total_stress(graph, pos1)
            deg = nx.degree(graph, graph.nodes)
            exp_factor_norm = expansion_factor_norm(pos0_arr, pos1_arr)
            edge_cross_norm = edge_crossings_norm(
                cross0-cross1, len(graph_copy.edges))
            d1 = distance_matrix(graph_copy)
            grad_diff = np.linalg.norm(gradient_kamada_kawai(
                pos0_arr, d0)-gradient_kamada_kawai(pos1_arr, d1))
            # Extra attributes
            max_deg = max(deg[n1], deg[n2])
            min_deg = min(deg[n1], deg[n2])
            sum_neighbour_deg_norm = sum_neighbour_degrees_norm(graph_copy, e)
            max_neighbour_deg_norm = max_neighbour_degrees_norm(graph_copy, e)
            max_jnc = max_j_node_centrality(graph_copy, pos1_arr, e)
            sum_jnc = sum_j_node_centrality(graph_copy, pos1_arr, e)
            nnodes, nedges = len(graph.nodes), len(graph.edges)
            graph_copy_entropy = graph_entropy_norm(graph_copy)
            # force in neighbourhood of n1 before relaxation
            force_after_1 = electro_forces_in_neighbourhood(
                graph_copy, n1, pos1, r)
            # force in neighbourhood of n2 before relaxation
            force_after_2 = electro_forces_in_neighbourhood(
                graph_copy, n2, pos1, r)
            force_after = force_after_1 + force_after_2
            cos_force_diff = cos_force_diff_in_neighbourhood(
                force_before, force_after)
            feature_to_var = {'graph_id': idx_graph, 'edge_id': idx_edge, 'num_nodes': nnodes, 'num_edges': nedges, 'edge_betweenness': eb[e], 'stress': st[e], 'max_deg': max_deg, 'min_deg': min_deg, 'is_bridge': e in bridges,
                              'diff_stress': total_stress0 - total_stress1, 'diff_cross': cross0 - cross1, 'diff_edglength': edgel0 - edgel1,
                              'benchmark': bench, 'exp_factor_norm': exp_factor_norm, 'edge_cross_norm': edge_cross_norm, 'sum_neighbour_deg_norm': sum_neighbour_deg_norm, 'max_neighbour_deg_norm': max_neighbour_deg_norm, 'max_jnc': max_jnc, 'sum_jnc': sum_jnc, 'diff_graph_entropy_norm': graph_copy_entropy-graph_entropy, 'cos_force_diff': cos_force_diff, 'diff_force': force_before-force_after}
            row = []
            for feature in list_features:
                if feature in feature_to_var.keys():
                    row.append(feature_to_var[feature])

            # row = [idx_graph, idx_edge, nnodes, nedges, eb[e], st[e], max_deg, min_deg, e in bridges,
            #       total_stress0 - total_stress1, cross0 - cross1, edgel0 - edgel1,
            #       bench, exp_factor_norm, edge_cross_norm, sum_neighbour_deg_norm, max_neighbour_deg_norm, max_jnc, sum_jnc, graph_copy_entropy-graph_entropy, cos_force_diff, force_before-force_after]
    data.append(row)
    return data


def generate_df(list_features: list, draw_f):
    """
    Generates a dataframe with the features specified in the paper.

    Returns:
        df (pd.DataFrame): dataframe with the features specified in the paper.
    """
    for bench in benchmarks:
        list_graphs = read_list_of_graphs(f'../data/{bench}/', 'graphml')
        data = []
        data.extend(generate_data_from_list(
            list_graphs, bench, list_features, draw_f))
        # cols = ['graph_id', 'edge_id', 'num_nodes', 'num_edges', 'edge_betweenness', 'stress', 'max_deg', 'min_deg', 'is_bridge', 'diff_stress', 'diff_cross', 'diff_edgelength', 'benchmark',
        #        'exp_factor_norm', 'edge_cross_norm', 'sum_neighbour_deg_norm', 'max_neighbour_deg_norm', 'max_jnc', 'sum_jnc', 'diff_graph_entropy_norm', 'cos_force_diff', 'diff_force']
        cols = list_features
        df = pd.DataFrame(data, columns=cols)
    return df


def plot_statistics(df):
    pd.plotting.scatter_matrix(
        df[['edge_betweenness', 'stress', 'diff_stress', 'diff_cross', 'diff_edgelength']])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    x = df[df['diff_cross'] < 0].stress
    y = df[df['diff_cross'] >= 0].stress
    ax1.hist(x, bins=10, alpha=0.5, label='diff_cross < 0')
    ax1.hist(y, bins=10, alpha=0.5, label='diff_cross >= 0')
    ax1.legend(loc='upper left')
    ax1.set_title('Stress')

    x = df[df['diff_cross'] < 0].edge_betweenness
    y = df[df['diff_cross'] >= 0].edge_betweenness

    ax2.hist(x, bins=10, alpha=0.5, label='diff_cross < 0')
    ax2.hist(y, bins=10, alpha=0.5, label='diff_cross >= 0')
    ax2.legend(loc='upper right')
    ax2.set_title('Edge betweenness')
    plt.show()


all_features = ['graph_id', 'edge_id', 'num_nodes', 'num_edges', 'edge_betweenness', 'stress', 'max_deg', 'min_deg', 'is_bridge',
                'diff_stress', 'diff_cross', 'diff_edglength',
                'benchmark', 'exp_factor_norm', 'edge_cross_norm', 'sum_neighbour_deg_norm', 'max_neighbour_deg_norm', 'max_jnc', 'sum_jnc', 'diff_graph_entropy_norm', 'cos_force_diff', 'diff_force']


def mainDataGen():
    list_features = all_features
    alg_name = 'kk'
    df = generate_df(list_features, alg_name)
    filename = 'graph_train_experiment_'+alg_name
    df.to_csv('../data/'+filename+'.csv', index=False)
    # plot_statistics(df)
