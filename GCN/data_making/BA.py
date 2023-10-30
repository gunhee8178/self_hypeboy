import torch
import random
import pickle
import argparse


from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_isolated_nodes
import numpy as np
import torch.distributions as dist

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ba_shapes')
parser.add_argument('--size', default=0, type=int)
parser.add_argument('--numbering', default=0)
args = parser.parse_args()

def transform(data):
    edge_index, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=data.x.shape[0])
    return Data(x=data.x[mask], edge_index=edge_index, y=data.y[mask])

num_motifs = 80
num_nodes = 300

# data = ExplainerDataset(
#     graph_generator=BAGraph(num_nodes=num_nodes, num_edges=5),
#     motif_generator='house',
#     num_motifs=num_motifs,
# )[0]
'''
data.x = torch.ones((data.y.shape[0], 10))
total_node_len = int(data.y.size()[0])
test_ratio = 0.20
perm = torch.randperm(total_node_len)

train_mask = torch.zeros(total_node_len, dtype=torch.bool)
test_mask = torch.zeros(total_node_len, dtype=torch.bool)
test_mask[perm[:int(total_node_len*test_ratio)]] = True
train_mask[perm[int(total_node_len*test_ratio):]] = True

data.train_mask = train_mask
data.test_mask = test_mask
'''

# with open(f'../dataset/{args.dataset}/ba_shapes_base_v1.pkl', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
# with open(f'../dataset/ba_shapes/ba_shapes.pkl', 'rb') as fin:
#     data = pickle.load(fin)

# print(data.y)
# ood_graph = BAGraph(num_nodes=num_nodes, num_edges=5)
# ood_graph.x = torch.rand(ood_graph.x.shape)
# data = Data(edge_index=data.edge_index, y=data.y, edge_mask=data.edge_mask, node_mask=data.node_mask)
'''
data_add_ood = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=5),
    motif_generator=CustomMotif(data),
    num_motifs=1,
)[0]

with open(f'../dataset/{args.dataset}/ba_shapes_ood_v1.pkl', 'wb') as f:
    pickle.dump(data_add_ood, f, pickle.HIGHEST_PROTOCOL)
'''


with open(f'../../dataset/ba_shapes/ba_shapes.pkl', 'rb') as fin:
    data = pickle.load(fin)

adj = to_dense_adj(edge_index=data.edge_index).squeeze()
edge_index = dense_to_sparse((adj + adj.t())/2)

print(data.edge_index.shape)
print(edge_index)
print((((adj + adj.t())/2 == adj)==False).sum())

# exit()

e = 1


sizes = [160]
# versions = [('v5', 77)]
versions = [('v4', 42), ('v5', 66), ('v6', 26), ('v7', 5)]
# random_mu = [0.0] * 8
# random_sigma = [1.0] * 8
# mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
#
# mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
#
# feat1 = np.random.multivariate_normal(mu_1, np.diag(sigma_1), 700)
#
# feat2 = np.random.multivariate_normal(mu_2, np.diag(sigma_2), 160)

# new_node = torch.ones(sizes, 10)

# with open(f'../../dataset/ba_shapes/ba_shapes_{0}_{"v5"}.pkl', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

ori_x = data.x
ori_y = data.y
ori_edge_index = data.edge_index
ori_train_mask = data.train_mask
ori_test_mask = data.test_mask
ori_node_mask = data.node_mask
ori_edge_mask = data.edge_mask
print (data)
for size in sizes:
    for version, seed in versions:
        torch.manual_seed(seed)
        random.seed(seed)

        # Generate the tensor

        # new_node = torch.ones(n, 10)

        new_y = torch.full((size,), 4)

        new_node_mask = torch.full((size,), False)
        new_train_mask = torch.full((size,), False)
        new_test_mask = torch.full((size,), False)

        edge_index = torch.zeros((2, size), dtype=torch.int64)
        for col in range(0, size, 2):
            '''
            probs = torch.tensor([300/(380 + size), 80/(380 + size), size/(380 + size)])  # 0~299, 300~699 순서대로 확률을 설정합니다.
            categorical = dist.Categorical(probs)

            # 샘플링
            sample = categorical.sample()

            num1 = col // e + ori_x.shape[0]
            if sample == 0:
                num2 = random.randint(0, 299)
            elif sample == 1:
                num2 = 300 + (random.randint(0, num_motifs - 1) * 5) + random.randint(0, 4)
            else:
                num2 = random.randint(700, 700+ size -1)

            while num1 == num2:
                if sample == 0:
                    num2 = random.randint(0, 299)
                elif sample == 1:
                    num2 = 300 + (random.randint(0, num_motifs - 1) * 5) + random.randint(0, 4)
                else:
                    num2 = random.randint(700, 700 + size - 1)
            '''
            num1 = col + ori_x.shape[0]
            edge_index[0, col] = num1
            edge_index[1, col] = 300 + (5 * (col // 2)) + random.randint(0, 5)

            edge_index[0, col+1] = num1+1
            edge_index[1, col+1] = random.randint(0, 300)

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



        data.x = torch.cat((ori_x, new_node))
        with open(f'../../dataset/ba_shapes/ba_shapes_{size}_{version}.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(data)

        # data.x = torch.Tensor(np.random.multivariate_normal(mu_1, np.diag(sigma_1), 860))
        # with open(f'../../dataset/ba_shapes/ba_shapes_{size}_{version}_2.pkl', 'wb') as f:
        #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        # print (data)
# with open(f'../../dataset/ba_shapes/ba_shapes_300_v0.pkl', 'rb') as fin:
#     data = pickle.load(fin)
# print(data)