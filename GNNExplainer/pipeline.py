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
from gnn.gcn_syn2 import GCN as GCN_syn


from gnnexplainer import GNNExplainer
from torch_geometric.data import Data

from metrics import top_k_fidelity, top_k_sparsity, top_k_accuracy, auc, top_k_recall

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ba_shapes_160_v0')
parser.add_argument('--date', default='0731')
parser.add_argument('--version', default='v1_1')
parser.add_argument('--text', default='v1')
parser.add_argument('--ood', default=0, type=int)
parser.add_argument('--hidden', default=100, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--device', default=3, type=int, help='CPU or GPU.')
parser.add_argument('--topk', default=6, type=int)

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--delta', default=1.0, type=float)
parser.add_argument('--gamma', default=0.05, type=float)
parser.add_argument('--score', default='avg', type=str)
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
            data = pickle.load(fin).cpu()
            nclass = 4
    elif 'community' in args.dataset.lower():
        with open(f'../dataset/BA_Community/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).cpu()
            nclass = 8
    elif 'tree' in args.dataset.lower():
        with open(f'../dataset/Tree/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).cpu()
            nclass = 2

    features = torch.Tensor(data.x).squeeze()
    # print(data.x.shape, data.edge_index.max())
    # exit()
    node_indices = torch.where(data.node_mask)[0].tolist()

    # 바꿀 부분
    gnnNets = GCN_syn(nfeat=features.squeeze().shape[1], nhid=args.hidden, nclass=nclass, device=device, dropout=0.5)
    if "ba_shapes" in args.dataset.lower():
        gnnNets.load_state_dict(torch.load("./models/0816/gcn_ba_shapes_epochs1000_lr0.01_v1.pt", map_location='cpu'))
        nclass=4

    elif 'community' in args.dataset.lower():
        gnnNets.load_state_dict(torch.load("./models/0808/gcn_BA_Community_epochs1000_lr0.003_v1.pt", map_location='cpu'))
        nclass=8

    elif 'cycle' in args.dataset:
        gnnNets.load_state_dict(torch.load("./models/0805/gcn_Tree_cycle_epochs1000_lr0.0001_v1.pt", map_location='cpu'))

    elif 'grid' in args.dataset:
        gnnNets.load_state_dict(torch.load("./models/0805/gcn_Tree_grid_epochs1000_lr0.0005_v1.pt", map_location='cpu'))


    gnnNets.to(device)
    gnnNets.eval()
    data.to(device)


    save_dir = os.path.join(f'./results/{args.date}',
                                         f"{args.dataset}",
                                         f"beta_{args.beta}",
                                         f"{args.version}_"
                                         f"gnnexplainer")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    gnnexplainer = GNNExplainer(gnnNets, epochs=args.epochs, lr=args.lr, nclass=nclass, args=args)



    duration = 0.0

    fidelity_score_list = []
    sparsity_score_list = []

    auc_score_list = []
    acc_score_list = []
    recall_score_list = []

    total_ood_len = 0
    total_select_len = 0
    exc = 0

    if 'shape' in args.dataset.lower():
        ood = [4]
    elif 'community' in args.dataset.lower():
        ood = [8]
    elif 'tree' in args.dataset.lower():
        ood = [2]

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    exc_a = 0
    exc_b = 0
    _, pred = gnnNets(data)[1].max(dim=1)

    ox1, ox2, ox3, ologits, oprob = gnnNets(data.to(device))
    _, opred = oprob.max(dim=1)
    for ori_node_idx in tqdm(range(400,700,1)):
        tic = time.perf_counter()
        if glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt")):
            file = glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt"))[0]
            edge_mask, x, edge_index, y, subset, edge_reals = torch.load(file)
            edge_mask = torch.from_numpy(edge_mask)
            node_idx = int(torch.where(subset == ori_node_idx)[0])

            sub_data = Data(x, edge_index).to(device)
            _, pred = gnnNets(sub_data)[1].max(dim=1)
            pred_label = pred[node_idx]

        else:
            data.to(device)
            x, edge_index, y, subset, kwargs = \
                gnnexplainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y, edge_reals=data.edge_mask)
            node_idx = int(torch.where(subset == ori_node_idx)[0])


            sub_data = Data(x, edge_index).to(device)
            gnnNets.eval()
            x1, x2, x3, logits, prob = gnnNets(sub_data)



            _, pred = prob.max(dim=1)
            pred_label = pred[node_idx]
            # edge_mask = gnnexplainer(x, edge_index, node_idx=node_idx, ori_label=pred_label)

            # save_path = os.path.join(save_dir, f"node_{ori_node_idx}.pt")
            # edge_mask = edge_mask.cpu()
            # edge_reals = kwargs['edge_reals'].cpu()
            # cache_list = [edge_mask.numpy(), x.cpu(), edge_index.cpu(), y.cpu(), subset.cpu(), edge_reals]
            # torch.save(cache_list, save_path)
            # sub_data.detach().cpu()

        # if (x1 - ox1[subset] > 0.001).sum() :
        #     print(ori_node_idx)
        #     if ori_node_idx == 699 :
        #         for q in range(len(subset)):
        #             print(x1[q] - ox1[subset[q]] < 0.001)
        #             if q ==4 :
        #                 print(x1[q])
        #                 print(ox1[subset[q]])
        #
            if pred_label != opred[ori_node_idx]:
                exc_a += 1
        #     print(ori_node_idx)
        #     print(f"x2\n{x2[node_idx]}\n{ox2[ori_node_idx]}\n{x2[node_idx]-ox2[ori_node_idx] < 0.00001}")
        #     print(f"x3\n{x3[node_idx]}\n{ox3[ori_node_idx]}\n{x3[node_idx]-ox3[ori_node_idx] < 0.00001}\n{logits[node_idx]}\n{ologits[ori_node_idx]}")
        #     print(pred_label.item(), data.y[ori_node_idx].item())
        #     exc_a += 1

        duration += time.perf_counter() - tic
        continue

        try:
            if edge_reals.sum() != 12:
                print(ori_node_idx, edge_reals.sum())
                exit()
            auc_score = auc(edge_reals, edge_mask)
            auc_score_list.append(auc_score)

            acc_score = top_k_accuracy(edge_reals, edge_mask, top_k)
            acc_score_list.append(acc_score)

            recall_score = top_k_recall(edge_reals, edge_mask, top_k)
            recall_score_list.append(recall_score)

        except:
            print(ori_node_idx)
            exc_b += 1
            continue


    print(exc_a)
    auc_scores = torch.tensor(auc_score_list)
    acc_scores = torch.tensor(acc_score_list)
    recall_scores = torch.tensor(recall_score_list)

    # print(f"auc score: {auc_scores.mean().item():.4f}, "
    #           f"acc score: {acc_scores.mean().item():.4f}, "
    #           f"recall score: {recall_scores.mean().item()} ({exc_a} {exc_b})\n")
    return auc_scores, acc_scores, recall_scores, exc_a , exc_b





if __name__ == '__main__':
    top_k = args.topk
    auc_scores, acc_scores, recall_scores, exc_a, exc_b = pipeline_NC(top_k)
    print(f"auc score: {auc_scores.mean().item():.4f}, "
          f"acc score: {acc_scores.mean().item():.4f}, "
          f"recall score: {recall_scores.mean().item()} ({exc_a} {exc_b})")

    if not os.path.isdir(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}'):
        os.makedirs(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}')

    with open(f'./txt/{args.date}/{args.dataset}/beta_{args.beta}/{args.text}.txt', 'a+') as txt:
        txt.write(f"auc score: {auc_scores.mean().item():.4f}, "
                    f"acc score: {acc_scores.mean().item():.4f}, "
                    f"recall score: {recall_scores.mean().item()} ({exc_a} {exc_b})\n")
