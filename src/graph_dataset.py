import networkx as nx
import pandas as pd
import numpy as np
import requests
#import osmnx as ox
import os
import time
import re
from tqdm.notebook import tqdm
from scipy.spatial import Delaunay
from random import choice


def _dist(u:np.array, v:np.array) -> float:
    return np.linalg.norm(u-v)

def _normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def _standardize(v):
    return (v - np.mean(v)) / np.std(v)



class GraphDataset:
    """
    Dataset with multiple Graphs and Edge Attributes
    """

    def __init__(self, df_graphs = None, df_edges = None, last_id = 0):
        self.last_id = last_id
        if df_graphs is None or df_edges is None:
            self.df_graphs = pd.DataFrame(columns=["id", "name", "type", "nodes", "edges", "new_edges"])
            self.df_edges = pd.DataFrame(columns=["graph_id", "u", "v", "label"])
        else:
            self.df_graphs = df_graphs
            self.df_edges = df_edges


    @classmethod
    def fromFile(cls, src):
        """Imports a dataset from the filesystem

        Parameters
        ----------
        src : str
            Path where the folder is stored
        """
        df_graphs = pd.read_csv(f"{src}/graphs.csv", index_col=0)
        df_edges = pd.read_csv(f"{src}/edges.csv", index_col=0)
        last_id = max(df_graphs['id']) + 1
        
        return cls(df_graphs, df_edges, last_id = last_id)


    def _getStatsTxt(self):
        n = len(self.df_graphs.index)
        n_edges = np.sum(self.df_graphs['edges'])
        mean_nodes = np.mean(self.df_graphs['nodes'])
        mean_edges = np.mean(self.df_graphs['edges'])
        mean_new_edges = np.mean(self.df_graphs['new_edges'])

        txt = f'''
        DATASET PROPERTIES
        ------------------
        Total number of graphs: {n}
        Total number of edges: {n_edges}

        Mean number of nodes: {mean_nodes}
        Mean number of edges: {mean_edges}
        Mean number of added edges: {mean_new_edges}

        Generated with seed: {self.seed}
        '''

        return txt


    def addGraph(self, G: nx.Graph, labels: list = None, seed: int = 123, **kwargs) -> None:
        """Adds a graph to the dataset
        
        Parameters
        ----------
        G : nx.Graph
            The new graph to add

        labels : list
            List of labels for each edge (1: new, 0: not new)

        seed : int
            Random seed used for generating the graph

        name : str
            (optional) Graph name
        """

        self.seed = seed
        nodes, edges = len(G.nodes), len(G.edges)

        if labels is None:
            labels = [0 for _ in range(edges)]

        new_edges = int(np.sum(np.array(labels)))

        self.df_graphs = self.df_graphs.append({"id": self.last_id, "nodes":nodes, "edges":edges, "new_edges":new_edges, **kwargs},ignore_index=True)

        for idx, (u, v) in enumerate(list(G.edges)):
            self.df_edges = self.df_edges.append({"graph_id": self.last_id, "u":u, "v":v, "label":labels[idx]},ignore_index=True)

        self.last_id += 1


    def addMetric(self, name:str, metric:callable, standardize:bool = False) -> None:
        """Adds a given edge metric to the dataframe
        
        Parameters
        ----------
        name : str
            Name of the attribute
        
        metric : callable
            Function computing the metric. The function input must be a networkx graph and the output
            must be a dictionary with ordered edges as the key.

        """
        if name not in self.df_edges:
            vals = []
            for i in range(len(self.df_graphs.index)):
                g = self.getGraph(i)
                v = metric(g).values()
                vals.extend(v)
            if standardize:
                vals = _standardize(vals)
            self.df_edges[name] = vals


    def export(self, path=".", name="") -> None:
        """Export the dataset

        The result is a folder with two csv files, one with all the edges and another with
        the graphs information.

        Parameters
        ----------
        path : str
            Path to save the files
        """
        dst = f"{path}/graph_ds-{name}"
        os.mkdir(dst)
        self.df_graphs.to_csv(f"{dst}/graphs.csv")
        self.df_edges.to_csv(f"{dst}/edges.csv")

        txt = self._getStatsTxt()
        f = open(f"{dst}/info.txt", "w")
        f.write(txt)
        f.close()


    def getGraph(self, graph_id:int, original:bool=False) -> nx.Graph:
        """Returns the NetworkX graph given its id
        
        Parameters
        ----------
        graph_id : int
            Graph identifier

        original : bool
            If True, returns the original graph without the added edges
        """
        G = nx.Graph()

        nodes = self.df_graphs["nodes"][graph_id]
        us = list(self.df_edges[self.df_edges['graph_id'] == graph_id]['u'])
        vs = list(self.df_edges[self.df_edges['graph_id'] == graph_id]['v'])
        tg = list(self.df_edges[self.df_edges['graph_id'] == graph_id]['label'])

        G.add_nodes_from( list(range(int(nodes))) )
        G.add_edges_from( [(u, v) for u, v in zip(us, vs)] )

        if original:
            us2 = list(self.df_edges[(self.df_edges['graph_id'] == graph_id) & (self.df_edges['label'] == 1)]['u'])
            vs2 = list(self.df_edges[(self.df_edges['graph_id'] == graph_id) & (self.df_edges['label'] == 1)]['v'])
            G.remove_edges_from( [(u, v) for u, v in zip(us2, vs2)] )

        nx.set_edge_attributes(G, {(u,v):t for u,v,t in zip(us,vs,tg)}, "target")

        return G


    def __str__(self):
        return f"Graph Dataset ({len(self.df_graphs.index)} graphs)"


    def __len__(self) -> int:
        return len(self.df_graphs.index)

    
    def __getitem__(self, pos:int) -> nx.Graph:
        return self.getGraph(pos)



####
# GRAPH-MODIFYING FUNCTIONS
####

def addRandomEdges(graph: nx.Graph, nEdges: int) -> tuple:
    """ Adds random edges to a given graph """
    nodes = list(graph.nodes)
    n = len(nodes)
    edges = []
    for i in range(nEdges):
        newEdge = False
        while not newEdge:
            i_u, i_v = np.random.randint(0, n-1), np.random.randint(0, n-1)
            edge = (nodes[i_u], nodes[i_v])
            if edge not in graph.edges(data=False) and edge not in edges:
                newEdge = True
        edges.append(edge)
    g = graph.copy()
    g.add_edges_from(edges)
    return g, edges


def sampleBFS(graph: nx.Graph, depth:int, seed:int = None):
    g = graph.copy()
    nnodes = len(g.nodes)
    
    if seed is not None:
        np.random.seed(seed)
    
    init_node = np.random.choice(g.nodes)
    elist = list(nx.bfs_edges(g, source=init_node, depth_limit=depth))
    nlist = [item for sublist in elist for item in sublist]
    g_sample = g.subgraph(nlist)
    return g_sample, init_node


def sampleProbBFS(graph: nx.Graph, depth:int, p:float, seed:int = None):
    g = graph.copy()
    nnodes = len(g.nodes)
    
    if seed is not None:
        np.random.seed(seed)

    init_node = np.random.choice(g.nodes)
    visited = [-1 for _ in range(nnodes)]
    queue = []
    queue.append(init_node)
    visited[init_node] = 0

    nlist = []

    while queue:
        u = queue.pop(0)
        if visited[u] <= depth:
            nlist.append(u)
        else:
            pass

        for v in g[u]:
            if visited[v] == -1:
                if np.random.uniform(0, 1) < p:
                    queue.append(v)
                visited[v] = visited[u]+1
    
    g_sample = g.subgraph(nlist)
    return g_sample, init_node
