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

classifier = XGBClassifier()
classifier.load_model('xgb.bin')

def relax_just_one(g: nx.Graph, graph_id:int, draw_f, data: pd.DataFrame, model: XGBClassifier):
    """Relax just the best edge"""

    data = data[data.graph_id == graph_id]
    X, y = preprocess_data(data)
    proba = make_predictions(model, X, ret_proba = True)

    max_proba_idx = np.argmax(proba)
    max_proba_edge = list(g.edges)[max_proba_idx]

    g2 = g.copy()
    g2.remove_edges_from([max_proba_edge])
    
    compareGraphs(g, g, draw_f(g), draw_f(g2))
    


