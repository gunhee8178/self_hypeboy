"""
FileName: explainers.py
Description: Explainable methods' set
Time: 2020/8/4 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""
from typing import Any, Callable, List, Tuple, Union, Dict, Sequence

from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

from rdkit import Chem
import networkx as nx
from torch_geometric.utils import to_networkx, to_dense_adj
from utils import k_hop_subgraph_with_default_whole_graph
import time
import random
import numpy as np

EPS = 1e-15


class ExplainerBase(nn.Module):
    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False, nclass=4, args=None):
        super().__init__()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.nclass = nclass
        self.args = args

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))

    def __clear_masks__(self):
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    # def __subgraph__(self, node_idx, x, edge_index, **kwargs):
    #     num_nodes, num_edges = x.size(0), edge_index.size(1)
    #
    #     subset, edge_index, mapping, edge_mask = subgraph(
    #         node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
    #         num_nodes=num_nodes, flow=self.__flow__())
    #
    #     x = x[subset]
    #     for key, item in kwargs.items():
    #         if torch.is_tensor(item) and item.size(0) == num_nodes:
    #             item = item[subset]
    #         elif torch.is_tensor(item) and item.size(0) == num_edges:
    #             item = item[edge_mask]
    #         kwargs[key] = item
    #
    #     return x, edge_index, mapping, edge_mask, kwargs
    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.__num_hops__ + 1, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        y = y[subset]
        return x, edge_index, y, subset, kwargs

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False, nclass=4, args=None):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph, molecule, nclass, args)

        self.nclass = nclass
        self.alpha = args.alpha
        self.beta = args.beta
        self.delta = args.delta
        self.gamma = args.gamma
        self.score = args.score

    def loss_oodgat(self, gid, nclass, pred, edge_mask, edge_index, x3, max_ent):
        entropy = -pred * torch.log(pred)

        if torch.isinf(entropy).sum() + torch.isnan(entropy).sum():
            entropy[torch.isinf(entropy)] = 0.0
            entropy[torch.isnan(entropy)] = 0.0

        entropy = entropy.sum(-1)
        inv_ent = max_ent - entropy
        # entropy = (entropy.max() - entropy) / (entropy.max() - entropy.min())

        # print(pred[torch.where(torch.isnan(entropy))], entropy[torch.where(torch.isnan(entropy))], torch.where(torch.isnan(entropy)))
        row, col = edge_index
        if self.score == 'avg':
            entropy_weight = (inv_ent[row]+inv_ent[col])/2
        elif self.score == 'dist':
            entropy_weight = max_ent - abs(inv_ent[row] - inv_ent[col])

        cos_sim = F.cosine_similarity(entropy_weight, edge_mask, dim=-1)
        # cos_loss = -torch.log(cos_sim)
        cos_loss = -cos_sim

        if self.nclass >= 8:
            lim_ent = 1.5

        elif self.nclass >= 4:
            lim_ent = 0.7

        else:
            lim_ent = 0.4

        entropy_eps = torch.where(entropy > lim_ent, True, False)
        uniform_entropy = torch.mean(-torch.log(pred), dim=1)

        if torch.isinf(uniform_entropy).sum() + torch.isnan(uniform_entropy).sum():
            uniform_entropy[torch.isinf(uniform_entropy)] = 0.0
            uniform_entropy[torch.isnan(uniform_entropy)] = 0.0

            entropy_eps[torch.isnan(uniform_entropy)] = False
            entropy_eps[torch.isnan(uniform_entropy)] = False

            # print("inf", pred[torch.isinf(uniform_entropy)])
            # print("nan", pred[torch.isnan(uniform_entropy)])

        if entropy_eps.sum():
            uniform_entropy = uniform_entropy[entropy_eps].sum() / entropy_eps.sum()
        else:
            uniform_entropy = 0

        # print Error
        if (torch.isinf(cos_loss).sum()  + torch.isnan(cos_loss).sum() ):
            print(f'{gid}:\n'
                  f'\tcos_sim: {cos_sim}'
                  f'\tcos_loss: {cos_loss}'
                  f'\tuniform: {uniform_entropy}'
                  f'\teps: {entropy_eps.sum()} / {len(entropy)}'
                  f'\tent: {torch.isnan(entropy_weight).sum()}'
                  f'\tmask: {torch.isnan(edge_mask).sum()}')
            cos_loss = torch.tensor(0.0).to(self.device)

        return (cos_loss * self.delta + uniform_entropy * self.gamma)

    def __loss__(self, raw_preds, x_label):
        loss_fn = nn.CrossEntropyLoss()
        target = torch.zeros(raw_preds[self.node_idx].unsqueeze(0).shape).to(raw_preds.device)
        target[:, x_label] = 1
        loss = loss_fn(raw_preds[self.node_idx].unsqueeze(0), target)
        # print(f'label: { x_label}\n'
        #       f'loss: {loss}\n'
        #       f'pred: {raw_preds[self.node_idx].unsqueeze(0)}')
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> None:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        max_ent = torch.log(torch.tensor(self.nclass)).item()
        for epoch in range(1, self.epochs+1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x

            nodesize = h.shape[0]
            mask_adj = to_dense_adj(edge_index=edge_index.to(self.device), edge_attr=self.edge_mask.sigmoid(),
                                    max_num_nodes=nodesize).squeeze(0)
            # # set the symmetric edge weights
            sym_mask = (mask_adj + mask_adj.transpose(0, 1)) / 2
            sym_edge_mask = sym_mask[edge_index[0], edge_index[1]]

            data = Data(x=h, edge_index=edge_index, edge_weight=sym_edge_mask, **kwargs).to(self.model.device)
            x3, raw_preds, _ = self.model(data)


            loss_gnnex = self.__loss__(raw_preds, ex_label)
            loss_ood = self.loss_oodgat(self.node_idx, self.nclass, raw_preds, sym_edge_mask, edge_index, x3, max_ent=max_ent)

            if self.alpha:
                loss = loss_gnnex + loss_ood * self.beta
            else:
                loss = loss_gnnex

            if epoch % 20 == 0:
                # print(f'#D#Loss:{loss.item()}')
                print(f'Epoch: {epoch} | Loss: {loss.item()} | Loss_pge: {loss_gnnex.item()} | Loss_ood: {loss_ood.item()}')
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        nodesize = h.shape[0]
        mask_adj = to_dense_adj(edge_index=edge_index.to(self.device), edge_attr=self.edge_mask,
                                max_num_nodes=nodesize).squeeze(0)
        # # set the symmetric edge weights
        sym_mask = (mask_adj + mask_adj.transpose(0, 1)) / 2
        sym_edge_mask = sym_mask[edge_index[0], edge_index[1]]

        return sym_edge_mask.detach()

    def forward(self, x, edge_index, mask_features=False,
                positive=True, ori_label=0, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        # self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        node_idx = kwargs.get('node_idx')
        self.node_idx = node_idx

        # Assume the mask we will predict
        # labels = tuple(i for i in range(self.nclass))
        # ex_labels = tuple(torch.tensor([label]).to(self.model.device) for label in labels)

        # Calculate mask
        # print('#D#Masks calculate...')
        # for ex_label in ex_labels:
        ex_label = torch.tensor([ori_label]).to(self.device)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        edge_mask = self.gnn_explainer_alg(x, edge_index, ex_label)
            # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))

        self.__clear_masks__()

        return edge_mask




    def __repr__(self):
        return f'{self.__class__.__name__}()'


