# Notebook to generate data and save it in filename specified by input argument.
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.graph_utils import max_j_node_centrality, sum_j_node_centrality, j_node_centrality, graph_entropy_norm, electro_forces_in_neighbourhood, cos_force_diff_in_neighbourhood
from src.graph_utils import gradient_kamada_kawai, max_neighbour_degrees_norm, sum_neighbour_degrees_norm, expansion_factor_norm, edge_crossings_norm
from src.graph_utils import stress, total_stress, num_crossings, mean_edge_length, nodes_dict_to_array, distance_matrix
from src.graph_parser import parseGraphmlFile
from src.graph_dataset import GraphDataset
from fa.forceatlas2 import ForceAtlas2
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Generate Dataframe
n = 0
m = 5
benchmarks = ['random-dag', 'rome', 'north']


all_features = ['graph_id', 'edge_id', 'num_nodes', 'num_edges', 'edge_betweenness', 'stress', 'max_deg', 'min_deg', 'is_bridge',
                'diff_stress', 'diff_cross', 'diff_edglength',
                'benchmark', 'exp_factor_norm', 'edge_cross_norm', 'sum_neighbour_deg_norm', 'max_neighbour_deg_norm', 'max_jnc', 'sum_jnc', 'diff_graph_entropy_norm','grad_diff']


def draw_fa2(g,pos=None):
    """
    Draw graph using ForceAtlas2 algorithm given optional initial positions

    Args:
        g (nx.Graph): graph
        pos (dict): initial positions

    Returns:
        dict: positions of nodes
    """
    fa2 = ForceAtlas2()
    posTuple = fa2.forceatlas2_networkx_layout(G=g, pos=pos, iterations=100)
    for x in posTuple.keys():
        posTuple[x] = np.array(posTuple[x])
    return posTuple


def draw_kk(g,pos=None):
    """
    Draw graph using Kamada-Kawai algorithm given optional initial positions

    Args:
        g (nx.Graph): graph
        pos (dict): initial positions

    Returns:
        dict: positions of nodes
    """
    return nx.kamada_kawai_layout(g,pos=pos)

algo_dict = {'fa2': draw_fa2, 'kk': draw_kk}


def read_list_of_graphs(dir_name, ext):
    """
    Read all the graphs in a directory given extension of the files

    Args:
        dir_name (str): directory name
        ext (str): extension of the files

    Returns:
        list: list of graphs
    """
    list_graphs = [parseGraphmlFile(dir_name+f, weighted=False, directed=False)
                   for f in os.listdir(dir_name) if f.endswith('.' + ext)]
    return list_graphs



def graph_to_df(graph: nx.Graph, idx_graph:int, draw_f,bench:str,list_features:list=all_features, return_df: bool=True, include_labels:bool=True) -> pd.DataFrame or list:
    """
    Convert a graph to a dataframe with the specified features

    Args:
        graph (nx.Graph): graph
        idx_graph (int): indentifier of the graph
        draw_f (function): function to draw the graph
        bench (str): benchmark of the graph
        list_features (list, optional): list of features to include in the dataframe. Defaults to all_features.
        return_df (bool, optional): return dataframe or list of lists. Defaults to True.

    Returns:
        pd.DataFrame or list: dataframe or list of lists with the specified features
    """
    #  Run Spectral +  algorithm chosen
    pos0 = draw_f(graph, pos=nx.spectral_layout(graph))

    #  Compute general graph attributes
    eb = nx.edge_betweenness(graph)     # edge betweenness
    st = stress(graph, pos0)
    cross0 = 0            
    if include_labels:
        cross0 = num_crossings(graph, pos0)
    edgel0 = mean_edge_length(graph, pos0)
    total_stress0 = total_stress(graph, pos0)
    deg = nx.degree(graph, graph.nodes)
    bridges = nx.bridges(graph)
    d0 = distance_matrix(graph)
    graph_entropy = graph_entropy_norm(graph)
    pos0_arr = nodes_dict_to_array(pos0)

    data = []
    for idx_edge, e in enumerate(graph.edges):

        n1, n2 = e

        # New position removing edge
        graph_copy = graph.copy()
        graph_copy.remove_edges_from([e])
        pos1 = draw_f(graph_copy, pos=pos0)
        pos1_arr = nodes_dict_to_array(pos1)
        cross1 = 0
        if include_labels:
            cross1 = num_crossings(graph, pos1)
        edgel1 = mean_edge_length(graph, pos1)
        total_stress1 = total_stress(graph, pos1)
        deg = nx.degree(graph, graph.nodes)
        exp_factor_norm = expansion_factor_norm(pos0_arr, pos1_arr)
        edge_cross_norm = 0
        if include_labels:
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
        
        feature_to_var = {'graph_id': idx_graph, 'edge_id': idx_edge, 'num_nodes': nnodes, 'num_edges': nedges, 'edge_betweenness': eb[e], 'stress': st[e], 'max_deg': max_deg, 'min_deg': min_deg, 'is_bridge': e in bridges,
                              'diff_stress': total_stress0 - total_stress1, 'diff_cross': cross0 - cross1, 'diff_edglength': edgel0 - edgel1,
                              'benchmark': bench, 'exp_factor_norm': exp_factor_norm, 'edge_cross_norm': edge_cross_norm, 'sum_neighbour_deg_norm': sum_neighbour_deg_norm, 'max_neighbour_deg_norm': max_neighbour_deg_norm, 'max_jnc': max_jnc, 'sum_jnc': sum_jnc, 'diff_graph_entropy_norm': graph_copy_entropy-graph_entropy, 'grad_diff': grad_diff}
        row = []
        for feature in list_features:
            if feature in feature_to_var.keys():
                if feature =='edge_cross_norm' or feature == 'diff_cross':
                    if include_labels:
                        row.append(feature_to_var[feature])
                else:
                    row.append(feature_to_var[feature])
            else:
                print('Feature not included ', feature)
        
        data.append(row)

    if return_df:
        return pd.DataFrame(data, columns=list_features)
    else:
        return data

def generate_data_from_list(list_graphs: list, bench: str, list_features: list, draw_f, idx_start: int = 0):
    """
    Generates a list of rows for a dataframe from the graphs in list_graphs.

    Args:
        list_graphs (list): list of graphs.
        bench (str): name of the benchmark.

    Returns:
        data (list): list of rows for df from list_graphs
    """
    data = []
    for idx_graph, graph in enumerate(list_graphs[n:m]):
        graph_data = graph_to_df(graph, idx_graph+idx_start, draw_f, bench,list_features, return_df=False)
        data.extend(graph_data)

    return data


def generate_df(list_features: list, draw_f):
    """
    Generates a dataframe with the features specified in the paper.

    Returns:
        df (pd.DataFrame): dataframe with the features specified in the paper.
    """
    data = []
    last_id = 0
    for bench in benchmarks:
        list_graphs = read_list_of_graphs(f'../data/{bench}/', 'graphml')
        data.extend(generate_data_from_list(
            list_graphs, bench, list_features, draw_f, last_id))
        last_id = data[-1][0]+1
    return pd.DataFrame(data, columns=list_features)


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


def main_data_gen(alg_name: str = 'kk'):
    list_features = all_features
    draw_f = algo_dict[alg_name]
    df = generate_df(list_features, draw_f)
    filename = 'graph_train_experiment_'+alg_name
    df.to_csv('../data/'+filename+'.csv', index=False)
    # plot_statistics(df)


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main_data_gen('kk')
   main_data_gen('fa2')
