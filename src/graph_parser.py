"""
Graph Parser module
Author: Raúl Higueras (raul.higueras@estudiantat.upc.edu)

Set of functions to parse different file formats with graph information.

All functions follow the same structure:

Parameters
----------
src : str
    Source of the file

weighted : bool
    Wether the graph contains weight information or not

directed : bool
    Wether to represent edges as directed or undirected.

Returns
-------
graph : nx.Graph or nx.DiGraph
    A Graph or DiGraph representation of the file (in the networkx format)
"""

import networkx as nx
import pandas as pd
import re
import numpy as np
import requests
import zstandard as zstd


def _rl(f):  # readline
    return f.readline().strip()


def parseNetFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    g, stopword = None, ""
    if directed:
        g, stopword = nx.DiGraph(), "*Arcs"
    else:
        g, stopword = nx.Graph(), "*Edges"

    f = open(src)
    found_edges = False
    for line in f.readlines():
        line = line.strip()
        if not found_edges:
            if line == stopword:
                found_edges = True
            else:
                pass
        else:
            if line == "*Edges":
                break
            line = re.sub(' +', ' ', line)
            if weighted:
                u,v,w = line.split(' ')
            else:
                u,v = line.split(' ')
            g.add_edges_from([(u,v)])
            # TODO: Implement adding weights to edges
    return g


def parseGmlFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    g = nx.DiGraph() if directed else nx.Graph()

    f = open(src)
    line = f.readline().strip()
    while line != "graph":
        line = f.readline().strip()
    _ = f.readline() # [
    line = f.readline().strip()
    while line:
        if line == "node":
            [f.readline() for _ in range(4)]
        elif line == "edge":
            f.readline() # [
            u = f.readline().strip().split(" ")[1]
            v = f.readline().strip().split(" ")[1]
            if weighted:
                w = f.readline().strip().split(" ")[1]
                #TODO: add edge weights
            f.readline() # ]
            g.add_edges_from([(u,v)])
            
        line = f.readline().strip()
    
    return g


def parseEdgeListFile(src:str, weighted:bool, directed:bool, commentSymbol:str='#') -> nx.Graph:
    g = nx.DiGraph() if directed else nx.Graph()

    f = open(src)
    line = f.readline().strip()
    while line[0] == commentSymbol:
        line = f.readline().strip()
    
    line = f.readline().strip()
    while line:
        line = re.sub(' +', ' ', line)
        line = re.sub('\s', ' ', line)
        if weighted:
            u,v,w = line.split(' ')
        else:
            u,v = line.split(' ')
        g.add_edges_from([(u,v)])
        #TODO: add weights
        
        line = f.readline().strip()
    
    return g


def parseCSV(src:str, weighted:bool, directed:bool) -> nx.Graph:
    df = pd.read_csv(src)
    h = list(df.columns)
    g = nx.DiGraph() if directed else nx.Graph()
    return nx.from_pandas_edgelist(df, create_using=g, source=h[0], target=h[1])


def parseGraFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    g = nx.DiGraph() if directed else nx.Graph()

    f = open(src)
    nnodes = int(_rl(f))
    nedges = int(_rl(f))

    
    degrees = [int(deg) for deg in  _rl(f).split(" ")]
    adjlists = [int(u) for u in _rl(f).split(" ")]
    i = 0
    for u, deg in enumerate(degrees):
        edges = [(u, v) for v in adjlists[i:i+deg]]
        g.add_edges_from(edges)
        i += deg

    # TODO: Consider weighted option (?)

    # TODO: Guess meaning of the last line
    
    return g


def parseSymmetricMTXFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    #g = nx.DiGraph() if directed else nx.Graph()

    f = open(src)
    l = _rl(f)
    while l[0] == '%':
        l = _rl(f)
    
    print(l)
    nrow, ncol, entries = [int(x) for x in l.split(" ")]

    mat = np.zeros((nrow, ncol))

    for _ in range(entries):
        l = _rl(f)
        i, j = [int(x)-1 for x in l.split(" ")[:2]]
        # TODO: Implement weights
        mat[i, j] = mat[j, i] = 1

    g = nx.from_numpy_matrix(mat)

    return g


def parseGtFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    # NOTE: It only reads online files (no local)
    # TODO: Standardize src reading
    content = requests.get(src).content
    if src.split(".")[-1] == 'zst':
        with zstd.ZstdDecompressor().stream_reader(content) as r:
            content = r.read()
    i = 0 # byte position
    def _rb(b, nbytes, res_type=None, endianess=None): #readBytes
        nonlocal i
        data = b[i:i+nbytes]
        i += nbytes
        if res_type == 'str':
            return data.decode()
        elif res_type == 'bool':
            return data == 1
        elif res_type == 'int':
            return int.from_bytes(data, endianess)
        
        return data

    G = nx.DiGraph() if directed else nx.Graph()
    _rb(content, 6, "str") # special init
    _rb(content, 1, "int", "little") # version
    end = 'big' if _rb(content, 1, "bool") else 'little' #endianess
    str_len = _rb(content, 8, "int", end) 
    comment = _rb(content, str_len, "str")
    directed = _rb(content, 1, "bool")
    N = _rb(content, 8, "int", end)
    G.add_nodes_from( list(range(N)) )
    d = 1 if N <= 2**8 else (2 if N <= 2**16 else (4 if N <= 2**32 else 8))
    for in_node in range(N):
        length = _rb(content, 8, "int", end)
        edges = [(in_node, _rb(content, d, "int", end)) for _ in range(length)]
        G.add_edges_from(edges)

    # TODO: process attributes
    
    return G

def parseGraphmlFile(src:str, weighted:bool, directed:bool) -> nx.Graph:
    # NOTE: it will return Graph or DiGraph depending on the specification 
    #       inside the file
    return nx.read_graphml(src)