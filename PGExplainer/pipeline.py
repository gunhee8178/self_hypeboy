import os
import glob
import time
import torch
import pickle
import numpy as np
import random
import sys
import argparse

from tqdm import tqdm
from gnn.gcn_ba_shapes import GCN as GCN_ba_shapes
from gnn.gcn_syn2 import GCN as GCN_syn

from pgexplainer import PGExplainer
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import roc_auc_score

from metrics import auc, top_k_recall, top_k_accuracy, top_k_ood

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora_full')
parser.add_argument('--date', default='0726')
parser.add_argument('--version', default='v1_1')
parser.add_argument('--text', default='v1')
parser.add_argument('--ood', default=0, type=int)
parser.add_argument('--hidden', default=500, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')
parser.add_argument('--topk', default=6, type=int)

parser.add_argument('--coff_size', default=0.05, type=float)
parser.add_argument('--coff_ent', default=1.0, type=float)

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--delta', default=1.0, type=float)
parser.add_argument('--gamma', default=5e-2, type=float)
parser.add_argument('--score', default='dist', type=str)
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def pipeline_NC(top_k):
    # data = Planetoid('../dataset', name='Cora', split='full')[0]
    if "ba_shapes" in args.dataset.lower():
        with open(f'../dataset/ba_shapes/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
            nclass = 4
    elif 'community' in args.dataset.lower():
        with open(f'../dataset/BA_Community/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
            nclass = 8
    elif 'cycle' in args.dataset.lower():
        with open(f'../dataset/Tree/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
            nclass = 2
    features = torch.Tensor(data["x"]).squeeze()

    # node_indice = torch.where(data.node_mask)[0].tolist()


    # 바꿀 부분
    if "ba_shapes" in args.dataset.lower():
        gnnNets = GCN_syn(nfeat=features.squeeze().shape[1], nhid=args.hidden, nclass=nclass, device=device,
                          dropout=0.0)
        gnnNets.load_state_dict(torch.load("./models/0914/gcn_ba_shapes_epochs1000_lr0.005_main_no_dropout.pt"))
        node_indice = range(300, 700, 5)

    elif 'community' in args.dataset.lower():
        gnnNets = GCN_syn(nfeat=features.squeeze().shape[1], nhid=args.hidden, nclass=nclass, device=device,
                          dropout=0.0)
        gnnNets.load_state_dict(torch.load("./models/0907/gcn_BA_Community_epochs1000_lr0.005_main_no_dropout.pt"))
        node_indice = range(300, 700, 5)

    elif 'cycle' in args.dataset.lower():
        gnnNets = GCN_syn(nfeat=features.squeeze().shape[1], nhid=args.hidden, nclass=nclass, device=device,
                          dropout=0.0)
        gnnNets.load_state_dict(torch.load("./models/0914/gcn_Tree_cycle_epochs1000_lr0.005_v3.pt"))
        node_indice = range(511, 991, 6)

    gnnNets = gnnNets.to(device)
    gnnNets.eval()

    save_dir = os.path.join(f'./results/{args.date}',
                                         f"{args.dataset}",
                                         f"beta_{args.beta}",
                                         f"{args.version}_"
                                         f"pgexplainer")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pgexplainer = PGExplainer(gnnNets, epochs=args.epochs, lr=args.lr, nclass=4, args=args, node_indice=node_indice)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    pgexplainer.get_explanation_network(data, is_graph_classification=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")

    duration = 0.0

    acc_score_list = []
    recall_score_list = []
    auc_score_list = []

    total_ood_len = []
    total_select_len = []
    exc = 0

    if 'shape' in args.dataset.lower():
        ood = [4]
    elif 'community' in args.dataset.lower():
        ood = [8]
    elif 'tree' in args.dataset.lower():
        ood = [2]


    for ori_node_idx in tqdm(node_indice):
        # print(ori_node_idx)
        tic = time.perf_counter()
        if glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt")):
            file = glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt"))[0]
            edge_mask, x, edge_index, y, subset, edge_reals = torch.load(file)
            edge_mask = torch.from_numpy(edge_mask)
            node_idx = int(torch.where(subset == ori_node_idx)[0])
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)

            sub_data = Data(x=x, edge_index=edge_index, y=y)
            sub_data.to(device)
            pgexplainer.__clear_masks__()
            _, pred_label2 = pgexplainer.model(sub_data)[2].max(-1)

        else:
            data.to(device)

            x, edge_index, y, subset, kwargs = \
                pgexplainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y, edge_reals=data.edge_mask)
            node_idx = int(torch.where(subset == ori_node_idx)[0])

            # gnnNets.eval()
            # sub_data = Data(x=x, edge_index=edge_index, y=y)
            # sub_data.to(device)


            edge_mask = pgexplainer.explain_edge_mask(x, edge_index)
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)


            save_path = os.path.join(save_dir, f"node_{ori_node_idx}.pt")
            edge_mask = edge_mask.cpu()
            edge_reals = kwargs['edge_reals'].cpu()
            cache_list = [edge_mask.numpy(), x.cpu(), edge_index.cpu(), y.cpu(), subset.cpu(), edge_reals]
            torch.save(cache_list, save_path)
            data.detach().cpu()

        if pred_label != y[node_idx]:
            exc += 1
            continue


        duration += time.perf_counter() - tic

        try:
            # print((y[edge_index]==4).any(axis=0))
            auc_score = auc(~(y[edge_index]==4).all(axis=0), edge_mask)

        except:
            print((y[edge_index]==4))
            print((y[edge_index]==4).any(axis=0))
            print()
            continue

        auc_score_list.append(auc_score)

        acc_score = top_k_accuracy(edge_reals, edge_mask, top_k)
        acc_score_list.append(acc_score)

        recall_score = top_k_recall(edge_reals, edge_mask, top_k)
        recall_score_list.append(recall_score)

        sub_data = Data(x=x, edge_index=edge_index, y=y)
        sub_data.to(device)
        edge_mask.to(device)

        selected_node_len, ood_node_len = top_k_ood(sub_data, edge_mask, top_k, undirected=True, ood=ood)
        total_select_len.append(selected_node_len)
        total_ood_len.append(ood_node_len)

    auc_scores = torch.tensor(auc_score_list)
    acc_scores = torch.tensor(acc_score_list)
    recall_scores = torch.tensor(recall_score_list)

    node_scores = torch.tensor(total_select_len).float()
    ood_scores = torch.tensor(total_ood_len).float()

    return auc_scores, acc_scores, recall_scores, node_scores, ood_scores, exc





if __name__ == '__main__':
    top_k = args.topk
    auc_scores, acc_scores, recall_scores, node_scores, ood_scores, exc = pipeline_NC(top_k)
    print(f"acc score: {auc_scores.mean().item():.4f}, "
          f"auc score: {acc_scores.mean().item():.4f}, "
          f"recall score: {recall_scores.mean().item():.4f} "
          f"ood score: {ood_scores.mean().item():.4f} / {node_scores.mean().item():.4f} {exc}")

    if not os.path.isdir(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}'):
        os.makedirs(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}')

    with open(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}/{args.text}.txt', 'a+') as txt:
        txt.write(  f"{' '.join(sys.argv)}\n"
                    f"acc score: {auc_scores.mean().item():.4f}, "
                    f"auc score: {acc_scores.mean().item():.4f}, "
                    f"recall score: {recall_scores.mean().item():.4f} "
                    f"ood score: {ood_scores.mean().item():.4f} / {node_scores.mean().item():.4f} {exc}"
                    f"\n\n")
