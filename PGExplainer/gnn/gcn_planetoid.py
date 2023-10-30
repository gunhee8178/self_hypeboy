import torch
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import GCNConv
from torch_geometric.typing import Adj, Size

seed = 66
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class GCNConv(GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConv, self).__init__(*args, **kwargs)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(GCN, self).__init__()
        self.device = device
        self.conv1 = GCNConv(nfeat, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nhid, normalize=False)
        self.conv3 = GCNConv(nhid, nhid, normalize=False)

        self.lin = nn.Linear(nhid, nclass)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(self.device)

        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = F.normalize(x1, p=2, dim=-1)
        x1 = self.dropout(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = F.normalize(x2, p=2, dim=-1)
        x2 = self.dropout(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = F.normalize(x3, p=2, dim=-1)
        x3 = self.dropout(x3)
        x3 = F.relu(x3)

        # emb = torch.cat([x1, x2, x3], dim=-1)
        emb = x3
        logits = self.lin(emb)

        # print(logits)
        return emb, logits, self.softmax(logits)


