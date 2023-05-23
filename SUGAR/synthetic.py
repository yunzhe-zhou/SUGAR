import numpy as np
from copy import deepcopy
from sklearn.preprocessing import PolynomialFeatures


# =================================================================================================================
# This contains the helper functions to generate the graph matrix for simulations
# =================================================================================================================


def generate_W(d=6, prob=0.5, low=0.5, high=2.0):
    """
    This function generates a random weighted adjaceecy matrix
    
    Input
    ----------
    d: number of nodes
    prob: prob of existing an edge
    low: the lower bound of the edge weight
    high: the upper bound of the edge weight
    
    Output
    ----------
    W: generated graph matrix
    """
    g_random = np.float32(np.random.rand(d,d)<prob)
    g_random = np.tril(g_random, -1)
    U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
    U[np.random.randn(d, d) < 0] *= -1
    W = (g_random != 0).astype(float) * U
    return W