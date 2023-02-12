import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tqdm import tqdm
import networkx as nx

from xgb_model import preprocess_data, make_predictions
from src.graph_utils import compareGraphs

df = pd.read_csv('../data/graph_train.csv')
print(df[['graph_id', 'benchmark']].head())
y = pd.cut(df['edge_cross_norm'], bins=[-1,-1e-10,1]).to_numpy()
df = df.drop(labels=['edge_cross_norm','edge_id','graph_id','num_nodes','num_edges','benchmark','max_deg','min_deg','Unnamed: 0'],axis=1)
list_columns = list(df)

for col in list_columns:
    print(col)
    if col!= 'edge_cross_norm' and col!= 'is_bridge':
        df[col] = pd.qcut(df[col],q=5)

X = df.to_numpy()

# Replace what is up there with preprocess data function.

# Have compare Graph output write the result in a file instead of showing it.

classifier = XGBClassifier()
classifier.load_model('xgb.bin')

#####
# Auxiliary functions 
#####
def bfs_on_edges(g: nx.Graph, edge: list|tuple, depth_limit) -> list:
    bfs_edges = {edge}
    sp = dict(nx.all_pairs_shortest_path_length(g))
    for e in nx.edge_bfs(g, edge[0]):
        if sp[e[0]][edge[0]] == depth_limit+1 or sp[e[1]][edge[0]] == depth_limit+1:
            break
        bfs_edges.add(e)
    
    for e in nx.edge_bfs(g, edge[1]):
        if sp[e[0]][edge[1]] == depth_limit+1 or sp[e[1]][edge[1]] == depth_limit+1:
            break
        bfs_edges.add(e)

    return list(bfs_edges)



#####
# Drawing functions 
#####
def relax_one(g: nx.Graph, data: pd.DataFrame, draw_f, model: XGBClassifier):
    """Relax just the best edge"""

    X, y = preprocess_data(data)
    proba = make_predictions(model, X, ret_proba = True)

    max_proba_idx = np.argmax(proba)
    max_proba_edge = list(g.edges)[max_proba_idx]

    g2 = g.copy()
    g2.remove_edges_from([max_proba_edge])
    
    # returns (num_crossings, aspect_ratio, mean_crossing_angle, pseudo_vertex_resolution, mean_angular_resolution, mean_edge_length, edge_length_variance)
    return compareGraphs(g, g, draw_f(g), draw_f(g2), show=False)


def just_relax(g: nx.Graph, data: pd.DataFrame, draw_f, model: XGBClassifier):
    """Relax just the best edge"""

    X, y = preprocess_data(data)
    proba = make_predictions(model, X, ret_proba = True)

    selected_idxs = np.where(proba>0.5)
    selected_edges = [list(g.edges)[idx] for idx in selected_idxs]

    g2 = g.copy()
    g2.remove_edges_from(selected_edges)
    
    # returns (num_crossings, aspect_ratio, mean_crossing_angle, pseudo_vertex_resolution, mean_angular_resolution, mean_edge_length, edge_length_variance)
    return compareGraphs(g, g, draw_f(g), draw_f(g2), show=False)


def relax_block(g: nx.Graph, data: pd.DataFrame, draw_f, model: XGBClassifier, depth_limit: int = 3):
    """Relax 1 edge -> block neighbours -> relax 1 edge -> block neighbours -> ...
    Note: no recomputing.
    """

    X, y = preprocess_data(data)
    proba = make_predictions(model, X, ret_proba = True)

    diff_crossings = -1
    relaxed_edges = []
    diff_crossings_hist = []

    g2 = g.copy()

    while diff_crossings < 0:
        max_proba_idx = np.argmax(proba)
        max_proba_edge = list(g.edges)[max_proba_idx]
        relaxed_edges.append(max_proba_edge)
        
        edges2block = bfs_on_edges(g, max_proba_edge, depth_limit)

        for e in [max_proba_edge, *edges2block]:
            proba[e] = -1

        g2.remove_edges_from([max_proba_edge])

        diff_crossings = compareGraphs(g, g, draw_f(g), draw_f(g2), show=False)[0]
        diff_crossings_hist.append(diff_crossings)
    
    min_crossings_idx = np.argmin(diff_crossings_hist)

    g2 = g.copy()
    g2.remove_edges_from([relaxed_edges[:min_crossings_idx]])

    # returns (num_crossings, aspect_ratio, mean_crossing_angle, pseudo_vertex_resolution, mean_angular_resolution, mean_edge_length, edge_length_variance)
    return compareGraphs(g, g, draw_f(g), draw_f(g2), show=False)



#####
# Evaluation functions 
#####

def eval_relax_one(model: XGBClassifier, df: pd.Dataframe, graphid2src: dict, draw_f):
    """Evaluate the method relax_one
    


