# -*- coding: utf-8 -*-
"""
This file including functions to generate different types of outliers given
the input dataset for benchmarking
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.data import Data

def gen_structural_outlier(data, m, n, p=0, directed=False, seed=None):
    """Generating structural outliers according to paper :
    cite:`ding2019deep`. We randomly select ``m`` nodes from the network
    and then make those nodes fully connected, and then all the ``m``
    nodes in the clique are regarded as outliers. We iteratively repeat
    this process until a number of ``n`` cliques are generated and the
    total number of structural outliers is ``m * n``.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    directed : bool, optional
        Whether the edges added are directed. Default: ``False``.
    seed : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0
        represents normal nodes.
    """
    if seed:
        torch.manual_seed(seed)

    new_edges = []

    outlier_idx = torch.randperm(data.x.shape[0])[:m * n]

    # connect all m nodes in each clique
    for i in range(n):
        new_edges.append(torch.combinations(outlier_idx[m * i: m * (i + 1)]))

    new_edges = torch.cat(new_edges)

    # drop edges with probability p
    if p != 0:
        indices = torch.randperm(len(new_edges))[:int((1-p) * len(new_edges))]
        new_edges = new_edges[indices]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    if not directed:
        new_edges = torch.cat([new_edges, new_edges.flip(1)], dim=0)

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier


def gen_contextual_outlier(data, n, k, seed=None):
    r"""Generating contextual outliers according to paper
    :cite:`ding2019deep`. We randomly select ``n`` nodes as the
    attribute perturbation candidates. For each selected node :math:`i`,
    we randomly pick another ``k`` nodes from the data and select the
    node :math:`j` whose attributes :math:`x_j` deviate the most from
    node :math:`i`'s attribute :math:`x_i` among ``k`` nodes by
    maximizing the Euclidean distance :math:`\| x_i − x_j \|`.
    Afterwards, we then substitute the attributes :math:`x_i` of node
    :math:`i` to :math:`x_j`.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.
    n : int
        Number of nodes converting to outliers.
    k : int
        Number of candidate nodes for each outlier node.
    seed : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The contextual outlier graph with modified node attributes.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0
        represents normal nodes.
    """
    if seed:
        torch.manual_seed(seed)

    outlier_idx = torch.randperm(data.x.shape[0])[:n]

    for i, idx in enumerate(outlier_idx):
        candidate_idx = torch.randperm(data.x.shape[0])[:k]
        euclidean_dist = torch.cdist(data.x[idx].unsqueeze(0), data.x[
            candidate_idx])

        max_dist_idx = torch.argmax(euclidean_dist, dim=1)
        max_dist_node = candidate_idx[max_dist_idx]
        data.x[idx] = data.x[max_dist_node]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    return data, y_outlier