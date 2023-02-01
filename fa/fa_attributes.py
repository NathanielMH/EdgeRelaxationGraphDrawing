import networkx as nx
import numpy as np


def charge_in_neighbourhood(G: nx.Graph, n1: int, layout: np.array, radius: float) -> np.array:
    """Returns the sum of charges in the neighbourhood of a node. CAN BE OPTIMIZED.
    Args:
        G (nx.Graph): Graph to be analyzed.
        node (int): Node to be analyzed.
        layout (np.array): Layout of the graph.
        radius (int): Radius of the neighbourhood.
    Returns:
        float: Sum of charges in the neighbourhood of a node.
    """
    charge = np.array([0, 0])
    m1 = 1 + np.count_nonzero(G[n1])
    for n2 in G.nodes():
        d = np.linalg.norm(layout[n1]-layout[n2])
        if d <= radius:
            m2 = 1 + np.count_nonzero(G[n2])
            charge += (layout[n1] - layout[n2])*m1*m2 / d**2
    return charge


def electro_forces_in_neighbourhood(G: nx.Graph, n1: int, layout: np.array, radius: float) -> float:
    """Returns the sum of electro forces in the neighbourhood of a node. CAN BE OPTIMIZED.
    Args:
        G (nx.Graph): Graph to be analyzed.
        node (int): Node to be analyzed.
        layout (np.array): Layout of the graph.
        radius (int): Radius of the neighbourhood.
    Returns:
        float: Sum of electro forces in the neighbourhood of a node.
    """
    force = 0.
    m1 = 1 + np.count_nonzero(G[n1])
    for n2 in G.nodes():
        d = np.linalg.norm(layout[n1]-layout[n2])
        if d <= radius:
            m2 = 1 + np.count_nonzero(G[n2])
            force += m1*m2*(layout[n1]-layout[n2]) / d**2
    return force


def cos_force_diff_in_neighbourhood(force_before: np.array, force_after: np.array) -> float:
    """Returns the cosine of the angle between two forces.
    Args:
        force_before (np.array): Force before relaxing.
        force_after (np.array): Force after relaxing.
    Returns:
        float: cosine of angle between two forces.
    """
    return np.dot(force_before, force_after)/(np.linalg.norm(force_before)*np.linalg.norm(force_after))

# Difference in norm, angle of sum of u,v charges (e = (u,v)) before and after relaxing e
