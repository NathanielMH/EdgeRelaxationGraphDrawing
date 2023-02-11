import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tqdm import tqdm
import networkx as nx

from xgb_model import preprocess_data, make_predictions
from src.graph_utils import compareGraphs
from src.graph_parser import parseGraphmlFile

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

classifier = XGBClassifier()
classifier.load_model('xgb.bin')

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
    selected_edges = [list(g.edges)[idx] for idx in max_proba_idxs]

    g2 = g.copy()
    g2.remove_edges_from(selected_edges)
    
    # returns (num_crossings, aspect_ratio, mean_crossing_angle, pseudo_vertex_resolution, mean_angular_resolution, mean_edge_length, edge_length_variance)
    return compareGraphs(g, g, draw_f(g), draw_f(g2), show=False)


def eval_relax_one(model: XGBClassifier, df: pd.Dataframe, graphid2src: dict, draw_f):
    """Evaluate the method relax_one
    
    Returns:
              avg_res: Vector of average difference in quality metrics
        perc_improved: % of improved graphs
    """

    graph_ids = set(df['graph_id'].values)

    res = []

    for graph_id in graph_ids:
        g = parseGraphmlFile(graphid2src[graph_id])
        data = df[df.graph_id == graph_id]
        res.append(relax_one(g, data, draw_f, model))
    
    res = np.asarray(res, float)
    avg_res = res.mean(axis=0)
    perc_improved = np.sum(res[0] < 0) / len(graph_ids) * 100

    return avg_res, perc_improved


def eval_just_relax(model: XGBClassifier, df: pd.Dataframe, graphid2src: dict, draw_f):
    """Evaluate the method just_relax
    
    Returns:
              avg_res: Vector of average difference in quality metrics
        perc_improved: % of improved graphs
    """

    graph_ids = set(df['graph_id'].values)

    res = []

    for graph_id in graph_ids:
        g = parseGraphmlFile(graphid2src[graph_id])
        data = df[df.graph_id == graph_id]
        res.append(just_relax(g, data, draw_f, model))
    
    res = np.asarray(res, float)
    avg_res = res.mean(axis=0)
    perc_improved = np.sum(res[0] < 0) / len(graph_ids) * 100

    return avg_res, perc_improved


