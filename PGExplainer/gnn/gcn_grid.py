import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, BatchNorm


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(GCN, self).__init__()
        self.device = device
        self.conv1 = GCNConv(nfeat, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nhid, normalize=False)
        self.conv3 = GCNConv(nhid, nhid, normalize=False)
        self.conv4 = GCNConv(nhid, nhid, normalize=False)

        self.lin = nn.Linear(nhid, nclass)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x1 = self.conv1(x, edge_index)
        x1 = F.normalize(x1, p=2, dim=-1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.normalize(x2, p=2, dim=-1)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = F.normalize(x3, p=2, dim=-1)
        x3 = F.relu(x3)

        x4 = self.conv4(x3, edge_index)
        x4 = F.normalize(x4, p=2, dim=-1)
        x4 = F.relu(x4)

        logits = self.lin(x4)

        return logits, self.softmax(logits), x4
