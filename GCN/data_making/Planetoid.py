import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_isolated_nodes
import argparse

def transform(data):
    edge_index, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=data.x.shape[0])
    return Data(x=data.x[mask], edge_index=edge_index, y=data.y[mask])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Citeseer')
args = parser.parse_args()

'''DATA LOAD'''
dataset = Planetoid('../dataset', name=args.dataset, split='full')
features = dataset[0]["x"].clone().detach().squeeze()
labels = dataset[0]["y"].clone().detach().squeeze()
edge_index = dataset[0]["edge_index"].clone().detach().squeeze()
train_mask = torch.Tensor(dataset[0]["train_mask"]).squeeze()
valid_mask = torch.Tensor(dataset[0]["val_mask"]).squeeze()
test_mask = torch.Tensor(dataset[0]["test_mask"]).squeeze()
# 행과 열 인덱스를 2차원으로 변환하여 선택
print(dataset[0])
print(train_mask.sum(), valid_mask.sum(), test_mask.sum())
# id_class = torch.tensor([2, 4, 5, 6])

if 'cora' in args.dataset.lower():
    id_class = torch.tensor([2, 4, 5, 6])
    mapping = {0: 4, 1: 4, 2: 0, 3: 4, 4: 1, 5: 2, 6: 3}

elif 'citeseer' in args.dataset.lower():
    id_class = torch.tensor([3, 4, 5])
    mapping = {0: 3, 1: 3, 2: 3, 3: 0, 4: 1, 5: 2}

elif 'pubmed' in args.dataset.lower():
    id_class = torch.tensor([0, 1])
    mapping = {0: 0, 1: 1, 2: 2}

node_mask = torch.isin(labels, id_class)
new_label = torch.zeros((features.shape[0],))
for i, value in enumerate(labels):
    new_label[i] = mapping[value.item()]

selected_train_mask = train_mask & node_mask
selected_valid_mask = valid_mask & node_mask
selected_test_mask = test_mask & node_mask


id_edge_index = edge_index[:, torch.all(node_mask[edge_index], 0)]

data = Data(
    x=features,
    y=labels,
    edge_index=edge_index,

    train_mask=selected_train_mask,
    valid_mask=selected_valid_mask,
    test_mask=selected_test_mask,

    node_mask=node_mask,
    id_class=new_label.long(),
    id_edge_index=id_edge_index
)

with open(f'../../dataset/{args.dataset}/{args.dataset}.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
print(data)