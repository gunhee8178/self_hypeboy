import torch
import random
import pickle
import argparse

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_isolated_nodes


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ba_shapes')
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
# Example usage:
num_nodes = 511
balanced_binary_tree_graph = generate_balanced_binary_tree(num_nodes)
print(balanced_binary_tree_graph)

num_motifs = 80
data = ExplainerDataset(
    graph_generator=CustomMotif(balanced_binary_tree_graph),
    motif_generator=CycleMotif(6),
    num_motifs=num_motifs,
)[0]

data.x = torch.ones(data.node_mask.shape[0], 10)
data.y = data.node_mask

total_node_len = data.node_mask.shape[0]
test_ratio = 0.20
perm = torch.randperm(total_node_len)

train_mask = torch.zeros(total_node_len, dtype=torch.bool)
test_mask = torch.zeros(total_node_len, dtype=torch.bool)

test_mask[perm[:int(total_node_len*test_ratio)]] = True
train_mask[perm[int(total_node_len*test_ratio):]] = True

data.train_mask = train_mask
data.test_mask = test_mask
with open(f'../../dataset/Tree/Tree_cycle.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
'''
with open(f'../../dataset/Tree/Tree_cycle.pkl', 'rb') as fin:
    data = pickle.load(fin)

e = 1

sizes = [240]
versions = [('v0', 42), ('v1', 66), ('v2', 26), ('v3', 5)]

num_motifs = 80
node_0 = 511

ori_x = data.x
ori_y = data.y
ori_edge_index = data.edge_index
ori_train_mask = data.train_mask
ori_test_mask = data.test_mask
ori_node_mask = data.node_mask
ori_edge_mask = data.edge_mask
print(data)
for size in sizes:
    for version, seed in versions:
        torch.manual_seed(seed)
        random.seed(seed)

        # Generate the tensor

        # new_node = torch.ones(n, 10)

        new_y = torch.full((size,), 2)

        new_node_mask = torch.full((size,), False)
        new_train_mask = torch.full((size,), False)
        new_test_mask = torch.full((size,), False)

        edge_index = torch.zeros((2, size), dtype=torch.int64)
        for col in range(0, size, 3):

            num1 = col + ori_x.shape[0]
            edge_index[0, col] = num1
            edge_index[1, col] = 511 + ((col // 3) * 6) + random.randint(0, 6)

            edge_index[0, col+1] = num1 + 1
            edge_index[1, col+1] = 511 + ((col // 3) * 6) + random.randint(0, 6)

            edge_index[0, col+2] = num1 + 2
            edge_index[1, col+2] = random.randint(0, node_0)


        adj = to_dense_adj(edge_index=edge_index).squeeze()

        edge_index = dense_to_sparse(adj+adj.t())[0]
        # print((adj.t()>2).sum(), (adj>2).sum(), ((adj+adj.t())>2).sum())
        # edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)

        print(edge_index.size())
        edge_mask = torch.zeros(edge_index.shape[1])



        data.y = torch.cat((ori_y, new_y))
        data.edge_index = torch.cat((ori_edge_index, edge_index), dim=1)
        data.train_mask = torch.cat((ori_train_mask, new_train_mask)).bool()
        data.test_mask = torch.cat((ori_test_mask, new_test_mask)).bool()
        data.node_mask = torch.cat((ori_node_mask, new_node_mask)).bool()
        data.edge_mask = torch.cat((ori_edge_mask, edge_mask)).bool()

        new_node = torch.ones(size, 10)
        data.x = torch.cat((ori_x, new_node), dim=0)
        with open(f'../../dataset/Tree/Tree_cycle_{size}_{version}.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(data)