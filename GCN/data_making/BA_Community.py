import torch
import random
import pickle
import argparse
from dgl.data import TreeCycleDataset


# from torch_geometric.datasets import ExplainerDataset
# from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
# from torch_geometric.datasets.graph_generator import BAGraph, GridGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_isolated_nodes
# from torch_geometric.utils import to_undirected

import torch.distributions as dist
import numpy as np
from dgl.data import BACommunityDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BA_Community')
parser.add_argument('--size', default=0, type=int)
parser.add_argument('--numbering', default=0)
args = parser.parse_args()


def generate_balanced_binary_tree(num_nodes):
    """
    Generate a balanced binary tree shape graph with 'num_nodes' nodes using PyTorch Geometric.

    Parameters:
        num_nodes (int): The number of nodes in the binary tree.

    Returns:
        Data: A PyTorch Geometric Data object representing the binary tree graph.
    """
    if num_nodes <= 0:
        raise ValueError("The number of nodes must be greater than zero.")

    # Calculate the depth of the binary tree
    depth = int.bit_length(num_nodes)  # Depth of the binary tree (1-indexed)

    # Generate the node features (in this example, we'll just use sequential indices as features)
    node_features = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)

    # Generate the edge indices to represent the binary tree structure
    edge_index = []
    for i in range(num_nodes):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < num_nodes:
            edge_index.append([i, left_child])
            edge_index.append([left_child, i])
        if right_child < num_nodes:
            edge_index.append([i, right_child])
            edge_index.append([right_child, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.zeros(num_nodes)
    # Create the PyG Data object representing the graph
    graph_data = Data(x=node_features, edge_index=edge_index, y=y)

    return graph_data

'''
dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=4),
    motif_generator='house',
    num_motifs=80,
    num_graphs=2,
)

print(dataset[0].edge_index.max())
print(dataset[1].edge_index.max())

dataset[1].edge_index += 700
dataset[1].y += 4


edge_index = torch.cat((dataset[0].edge_index, dataset[1].edge_index), dim=1)
edge_mask = torch.cat((dataset[0].edge_mask, dataset[1].edge_mask), dim=0)
y = torch.cat((dataset[0].y, dataset[1].y), dim=0)
node_mask = torch.cat((dataset[0].node_mask, dataset[1].node_mask), dim=0)


src = torch.randint(0, 699, (1, 350))
dst = torch.randint(700, 1399, (1, 350))

inter_edge = torch.cat((src, dst), dim=0)
adj = to_dense_adj(edge_index=inter_edge).squeeze()
# #
# inter_edge = dense_to_sparse(adj + adj.t())[0]
# inter_edge_mask = torch.zeros(inter_edge.shape[1], dtype=torch.bool)
#
# edge_index = torch.cat((edge_index, inter_edge), dim=1)
# edge_mask = torch.cat((edge_mask, inter_edge_mask), dim=0)

data = Data(y=y, edge_index=edge_index, edge_mask=edge_mask, node_mask=node_mask)

# feature generation
random_mu = [0.0] * 8
random_sigma = [1.0] * 8

mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
feat1 = np.random.multivariate_normal(mu_1, np.diag(sigma_1), 700)

mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
feat2 = np.random.multivariate_normal(mu_2, np.diag(sigma_2), 700)

data.x = torch.Tensor(np.concatenate([feat1, feat2]))
# data.y = data.node_mask

total_node_len = data.node_mask.shape[0]
test_ratio = 0.20
perm = torch.randperm(total_node_len)

train_mask = torch.zeros(total_node_len, dtype=torch.bool)
test_mask = torch.zeros(total_node_len, dtype=torch.bool)

test_mask[perm[:int(total_node_len*test_ratio)]] = True
train_mask[perm[int(total_node_len*test_ratio):]] = True

data.train_mask = train_mask
data.test_mask = test_mask
with open(f'../../dataset/BA_Community/BA_Community.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
'''

with open(f'../../dataset/BA_Community/BA_Community.pkl', 'rb') as fin:
    data = pickle.load(fin)
print(data)

e = 4

sizes = [320]

num_motifs = 80
node_0 = 300


random_mu = [0.0] * 8
random_sigma = [1.0] * 8
mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)

mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
print(data)
ori_x = data.x
ori_y = data.y
ori_edge_index = data.edge_index
ori_train_mask = data.train_mask
ori_test_mask = data.test_mask
ori_node_mask = data.node_mask
ori_edge_mask = data.edge_mask
versions = [('v4', 42), ('v5', 66), ('v6', 26), ('v7', 5)]
for size in sizes:
    for version, seed in versions:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        n = size

        # Generate the tensor

        # new_node = torch.ones(n, 10)

        new_y = torch.full((n,), 8)

        new_node_mask = torch.full((n,), False)
        new_train_mask = torch.full((n,), False)
        new_test_mask = torch.full((n,), False)

        edge_index = torch.zeros((2, n), dtype=torch.int64)

        for col in range(0, n, 2):

            num1 = col + ori_x.shape[0]
            if col < 160:
                edge_index[1, col] = 300 + (5 * (col // 2)) + random.randint(0, 5)
                edge_index[1, col+1] = random.randint(0, 300)
                # print(num1, edge_index[1, col])
            else:
                edge_index[1, col] = 1000 + (5 * ((col-160) // 2)) + random.randint(0, 5)
                edge_index[1, col+1] = random.randint(700, 1000)
            # print(edge_index[1, col+1])
            edge_index[0, col] = num1
            edge_index[0, col+1] = num1+1


        adj = to_dense_adj(edge_index=edge_index).squeeze()

        edge_index = dense_to_sparse(adj+adj.t())[0]
        # print((adj.t()>2).sum(), (adj>2).sum(), ((adj+adj.t())>2).sum())
        # edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)

        # print(edge_index.size())
        edge_mask = torch.zeros(edge_index.shape[1])



        data.y = torch.cat((ori_y, new_y))
        data.edge_index = torch.cat((ori_edge_index, edge_index), dim=1)
        data.train_mask = torch.cat((ori_train_mask, new_train_mask)).bool()
        data.test_mask = torch.cat((ori_test_mask, new_test_mask)).bool()
        data.node_mask = torch.cat((ori_node_mask, new_node_mask)).bool()
        data.edge_mask = torch.cat((ori_edge_mask, edge_mask)).bool()

        feat1 = np.random.multivariate_normal(mu_1, np.diag(sigma_1), size//2)

        feat2 = np.random.multivariate_normal(mu_2, np.diag(sigma_2), size//2)

        new_node = torch.Tensor(np.concatenate([feat1, feat2]))

        data.x = torch.cat((ori_x, new_node), dim=0)
        with open(f'../../dataset/BA_Community/BA_Community_{size}_{version}.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


