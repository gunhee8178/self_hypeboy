import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn.parameter import Parameter
import pandas as pd


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, device):
        super(GCN, self).__init__()
        self.device = device
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)
        self.bn1 = BatchNorm(nhid, track_running_stats=True, momentum=1.0)
        self.bn2 = BatchNorm(nhid, track_running_stats=True, momentum=1.0)
        self.bn3 = BatchNorm(nhid, track_running_stats=True, momentum=1.0)
        self.lin = nn.Linear(nhid, nclass)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)

        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = F.relu(x2)
        x2 = self.bn2(x2)

        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = self.bn3(x3)

        # x = self.lin1(torch.cat((x1, x2, x3), 1))
        x = self.lin(x3)
        return x, F.softmax(x, dim=1), x3

 
