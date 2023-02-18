from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.graph_parser import read_list_of_graphs, parseGraphmlFile
from src.graph_utils import compute_graph_metrics, compareGraphs

######
# AUX FUNCTIONS
######

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

def compareRelaxedDrawing(graph_id, bench, classifier, draw_f, thresh=0.5, num_edges=None, debug=False, blocked_edges=None, rm_edges=None, graph=None):

    if blocked_edges is None:
        blocked_edges = []
    
    if rm_edges is None:
        rm_edges = []
    if graph is None:
        ...
        #g = graph_lists[bench][graph_id]
    else:
        g = graph
    pos1 = draw_f(g)

    g_df = pd.read_csv('../data/graph_train.csv')
    g_df = g_df[(g_df.graph_id == graph_id) & (g_df.benchmark == bench)].reset_index(drop=True)
    g_df = g_df.drop(labels=['edge_cross_norm','edge_id','graph_id','num_nodes','num_edges','benchmark','max_deg','min_deg','Unnamed: 0','diff_cross'],axis=1)
    #print(g_df.columns)
    y_pred = classifier.predict_proba(g_df.to_numpy())[:, 1]
    if num_edges != None:
        edges_max_prob = np.argsort(y_pred)
        edges_max_prob = [e for e in edges_max_prob if (y_pred[e] > thresh and e not in blocked_edges)]
        edges2rm = edges_max_prob[-num_edges:]
    else:  
        edges2rm = np.where(y_pred>thresh)[0]
        edges2rm = [e for e in edges2rm if e not in blocked_edges]
    edges2rm.extend(rm_edges)
    g_copy = g.copy()
    g_copy.remove_edges_from([list(g.edges())[idx] for idx in edges2rm])

    pos2 = draw_f(g_copy)

    if debug:
        print(f"Removed edges: {len(edges2rm)}")

    return compareGraphs(g, g, pos1, pos2, show=debug, rmedges=[list(g.edges())[idx] for idx in edges2rm]), edges2rm


def validate_krelax(graphs, classifier, draw_f):
    #thresh_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    #thresh_labels = [f'thresh_{val}' for val in thresh_vals]
    values = []

    for graphs in tqdm(graphs):
        cross_val = compareRelaxedDrawing(graph_id, 'rome', classifier, draw_f, debug=False, thresh=0.5)[0][0]
        values.append(cross_val)

    return values


######
# MAIN FUNCTIONS
######


def read_input_graphs(graph_datasets: list[str], limit:int = np.Infinity):
    graphs = []
    for graph_ds in tqdm(graph_datasets):
        read_graphs = read_list_of_graphs(graph_ds, 'graphml', parseGraphmlFile)
        selected_graphs = np.random.choice(read_graphs, limit)
        graphs.extend([(g, graph_ds) for g in selected_graphs])
    return graphs


def create_dataset(graphs: list[tuple[nx.Graph, str]]):
    data = []
    draw_f = lambda g: nx.kamada_kawai_layout(g, pos=nx.spectral_layout(g))
    for idx, (g, bench) in tqdm(list(enumerate(graphs))):
        data.extend(compute_graph_metrics(g, draw_f, idx, bench))
    
    cols = ['graph_id', 'edge_id', 'num_nodes', 'num_edges', 'edge_betweenness', 'stress', 'max_deg', 'min_deg', 'is_bridge', 'diff_stress', 'diff_cross', 'diff_edgelength', 'benchmark', 'exp_factor_norm', 'edge_cross_norm', 'sum_neighbour_deg_norm', 'max_neighbour_deg_norm','max_jnc','sum_jnc']
    df = pd.DataFrame(data, columns=cols)
    return df

def divide_train_test(df: pd.DataFrame, graphs: list[nx.Graph], perc_test: float):
    graph_ids = set(df['graph_id'])
    num_graphs = len(graph_ids)
    test_ids = np.random.choice(list(graph_ids), int(num_graphs*perc_test/100))
    train_ids = graph_ids.difference(test_ids)

    df_train = df[df['graph_id'].isin(train_ids)]
    df_test = df[df['graph_id'].isin(test_ids)]

    df_train.to_csv('experiment_data/train.csv')
    df_test.to_csv('experiment_data/test.csv')

    graphs_train = [graphs[idx] for idx in train_ids]
    graphs_test = [graphs[idx] for idx in test_ids]

    return df_train, df_test, graphs_train, graphs_test

def train_models(df_train: pd.DataFrame, df_test: pd.DataFrame):

    df_train['is_bridge'] = df_train['is_bridge'].astype(float)
    df_test['is_bridge'] = df_test['is_bridge'].astype(float)

    y_train = (df_train['edge_cross_norm']>0).to_numpy(dtype=int)
    y_test = (df_test['edge_cross_norm']>0).to_numpy(dtype=int)

    X_train = df_train.drop(labels=['edge_cross_norm','edge_id','graph_id','num_nodes','num_edges','benchmark','max_deg','min_deg','diff_cross','is_bridge'],axis=1).to_numpy(dtype=float)
    X_test = df_test.drop(labels=['edge_cross_norm','edge_id','graph_id','num_nodes','num_edges','benchmark','max_deg','min_deg','diff_cross','is_bridge'],axis=1).to_numpy(dtype=float)

    T = 0.5
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train,verbose = 3)
    y_pred = xgb.predict_proba(X_test)
    y_pred_train = xgb.predict_proba(X_train)
    y_pred = np.array([1 if y[1]>T else 0 for y in y_pred])
    y_pred_train = np.array([1 if y[1]>T else 0 for y in y_pred_train])
    pos = np.sum([1 if y == 1 else 0 for y in y_test])
    f1 = f1_score(y_test,y_pred)
    print(f1)
    print('Amount of better:',pos)
    print('Amount of neutral or worse:',len(y_test)-pos)
    print('Accuracy in test:',(sum([1 if y_test[i]==y_pred[i] else 0 for i in range(len(y_pred))]))/len(y_test))
    print('Accuracy in train:',(sum([1 if y_train[i]==y_pred_train[i] else 0 for i in range(len(y_pred_train))]))/len(y_train))

    xgb.save_model("datamodel_train_full.json")

    return xgb

def evaluate_models():
    ...

def main():
    np.random.seed(1234)
    graph_datasets = ['data/north/', 'data/random-dag/']
    limit_graphs_per_dataset = 3
    perc_test = 20

    # Read input graphs
    print(f"Reading input graphs ({', '.join(graph_datasets)})...")
    graphs = read_input_graphs(graph_datasets, limit=limit_graphs_per_dataset)

    # Create dataset
    print(f"Creating dataset...")
    df = create_dataset(graphs)

    # Divide train-test
    print("Dividing dataset into train and test")
    df_train, df_test, graphs_train, graphs_test = divide_train_test(df, graphs, perc_test)

    # Train models
    model = train_models(df_train, df_test)

    # Evaluate models


main()