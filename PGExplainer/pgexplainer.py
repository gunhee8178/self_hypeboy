from typing import Optional
from math import sqrt

import os
import random
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx, to_dense_adj, k_hop_subgraph
from utils import k_hop_subgraph_with_default_whole_graph



EPS = 1e-6

class PGExplainer(nn.Module):
    def __init__(self, model, epochs: int = 20, lr: float = 0.003,
                 top_k: int = 6, args=None, nclass=4, node_indice = None, num_hops: Optional[int] = None):
        # lr=0.005, 0.003
        super(PGExplainer, self).__init__()
        # 내가 추가한 부분


        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k
        self.__num_hops__ = num_hops
        self.device = model.device


        self.coff_size = args.coff_size
        self.coff_ent = args.coff_ent
        self.init_bias = 0.0
        self.t0 = 5.0
        self.t1 = 2.0

        self.dataset = args.dataset
        self.nclass = nclass
        self.node_indice = node_indice
        self.alpha = args.alpha
        self.beta = args.beta
        self.delta = args.delta
        self.gamma = args.gamma
        self.score = args.score

        self.elayers = nn.ModuleList()
        # 바꿀 부분
        self.elayers.append(nn.Sequential(nn.Linear(args.hidden * 6, 64), nn.ReLU()))
        self.elayers.append(nn.Sequential(nn.Linear(64, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))

        self.elayers.to(self.device)
        self.ckpt_path = os.path.join(f'./checkpoint/{args.date}/{args.dataset}/beta_{args.beta}/'
                                      f'PGE_generator_{args.version}.pth')
        self.sub_path = os.path.join(f'./checkpoint/subgraph/{args.dataset}')

        if not os.path.isdir(os.path.join(f'./checkpoint/{args.date}/{args.dataset}/beta_{args.beta}')):
            os.makedirs(os.path.join(f'./checkpoint/{args.date}/{args.dataset}/beta_{args.beta}'))

        if not os.path.isdir(os.path.join(f'./checkpoint/subgraph/{args.dataset}')):
            os.makedirs(os.path.join(f'./checkpoint/subgraph/{args.dataset}'))

    def __set_masks__(self, x, edge_index, edge_mask=None, init="normal"):
        """ Set the weights for message passing """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        """ return the number of layers of GNN model """
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def tmp_edgemask(self, pred, edge_index,):
        entropy = -pred * torch.log(pred)

        # if torch.isinf(entropy).sum() + torch.isnan(entropy).sum():
        #     entropy[torch.isinf(entropy)] = 0.0
        #     entropy[torch.isnan(entropy)] = 0.0

        entropy = entropy.sum(-1)
        norm_entropy = ((entropy - entropy.mean()) / entropy.std()).sigmoid()
        row, col = edge_index
        if self.score == 'avg':
            entropy_weight = 1 - ((norm_entropy[row]+norm_entropy[col]) / 2)
        elif self.score == 'dist':
            entropy_weight = 1 - abs(norm_entropy[row] - norm_entropy[col])
        elif self.score == 'max':
            entropy_weight = 1 - torch.max(norm_entropy[row], norm_entropy[col])


        return entropy_weight

    def __loss__(self, prob, ori_pred):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit = prob[ori_pred].to(self.device)
        logit = logit + EPS
        # print(prob, ori_pred)
        pred_loss = -torch.log(logit)

        edge_mask = torch.sigmoid(self.mask_sigmoid)
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        if torch.isnan(loss):
            print(prob, size_loss, mask_ent_loss)
            print(loss)
        return loss

    def loss_oodgat(self, pred, edge_mask, edge_index,):
        entropy = -pred * torch.log(pred)

        # if torch.isinf(entropy).sum() + torch.isnan(entropy).sum():
        #     entropy[torch.isinf(entropy)] = 0.0
        #     entropy[torch.isnan(entropy)] = 0.0

        entropy = entropy.sum(-1)
        norm_entropy = ((entropy - entropy.mean()) / entropy.std()).sigmoid()
        # norm_entropy = entropy
        # entropy = (entropy.max() - entropy) / (entropy.max() - entropy.min())

        # print(pred[torch.where(torch.isnan(entropy))], entropy[torch.where(torch.isnan(entropy))], torch.where(torch.isnan(entropy)))
        row, col = edge_index
        if self.score == 'avg':
            entropy_weight = 1 - ((norm_entropy[row]+norm_entropy[col]) / 2)
        elif self.score == 'dist':
            entropy_weight = 1 - abs(norm_entropy[row] - norm_entropy[col])
        elif self.score == 'max':
            entropy_weight = 1 - torch.max(norm_entropy[row], norm_entropy[col])

        cos_sim = F.cosine_similarity(entropy_weight, edge_mask, dim=-1)

        '''
        print("row", y_dict[row])
        print("row", norm_entropy[row])
        print("row", entropy[row])

        print("col", y_dict[col])
        print("col", norm_entropy[col])
        print("col", entropy[col])

        print("ent", entropy_weight)
        '''

        # cos_loss = -torch.log(cos_sim)
        cos_loss = -cos_sim

        if self.nclass >= 8:
            lim_ent = 1.6

        elif self.nclass >= 4:
            lim_ent = 0.9

        else:
            lim_ent = 0.6

        entropy_eps = torch.where(norm_entropy > lim_ent, True, False)
        uniform_entropy = torch.mean(-torch.log(pred), dim=1)

        '''
        if torch.isinf(uniform_entropy).sum() + torch.isnan(uniform_entropy).sum():
            uniform_entropy[torch.isinf(uniform_entropy)] = 0.0
            uniform_entropy[torch.isnan(uniform_entropy)] = 0.0

            entropy_eps[torch.isnan(uniform_entropy)] = False
            entropy_eps[torch.isnan(uniform_entropy)] = False
        '''
            # print("inf", pred[torch.isinf(uniform_entropy)])
            # print("nan", pred[torch.isnan(uniform_entropy)])

        if entropy_eps.sum():
            uniform_entropy = uniform_entropy[entropy_eps].mean()
        else:
            uniform_entropy = 0

        # print Error
        if (torch.isinf(cos_loss).sum()  + torch.isnan(cos_loss).sum() ):
            '''
            print(f'{gid}:\n'
                  f'\tcos_sim: {cos_sim}'
                  f'\tcos_loss: {cos_loss}'
                  f'\tuniform: {uniform_entropy}'
                  f'\teps: {entropy_eps.sum()} / {len(entropy)}'
                  f'\tent: {torch.isnan(entropy_weight).sum()}'
                  f'\tmask: {torch.isnan(edge_mask).sum()}')   
            '''

            cos_loss = torch.tensor(0.0).to(self.device)
        # print(f'cos_loss: {cos_loss} uniform_entropy: {uniform_entropy}')
        return (cos_loss * self.delta + uniform_entropy * self.gamma)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape).to(log_alpha.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None):
        x, embed, edge_index, tmp = inputs
        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        embed = embed.to(self.device)
        # edge_index = edge_index.to(self.device)

        col, row = edge_index
        f1 = embed[col]
        f2 = embed[row]

        # using the node embedding to calculate the edge weight
        h = torch.cat([f1, f2], dim=-1)
        for elayer in self.elayers:
            h = elayer(h)

        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.mask_sigmoid = values
        #
        mask_adj = to_dense_adj(edge_index=edge_index.to(values.device), edge_attr=values, max_num_nodes=nodesize).squeeze(0)
        # # set the symmetric edge weights
        sym_mask = (mask_adj + mask_adj.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)
        # the model prediction with edge mask
        data = Batch.from_data_list([Data(x=x.to(self.device), edge_index=edge_index.to(self.device)).to(self.device)])
        data = data.to(self.device)

        emb, logits, prob = self.model(data)
        self.__clear_masks__()
        return prob.squeeze(), emb.squeeze(), edge_mask

    def get_model_output(self, x, edge_index, edge_mask=None, **kwargs):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))

        with torch.no_grad():
            data = Data(x=x, edge_index=edge_index)
            data.to(self.device)
            outputs = self.model(data)
        self.__clear_masks__()
        return outputs


    def get_explanation_network(self, dataset, is_graph_classification=True):
        if os.path.isfile(self.ckpt_path):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self.ckpt_path)
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        elif not is_graph_classification:
            self.train_NC_explanation_network(dataset)

    def eval_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_mask: torch.Tensor=None, **kwargs) -> None:
        _, _, prob = self.get_model_output(x, edge_index, edge_mask=edge_mask)
        return prob.squeeze()

    def explain_edge_mask(self, x, edge_index, **kwargs):
        # data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        # data.to(self.device)
        with torch.no_grad():
            emb, _, prob = self.get_model_output(x, edge_index)
            # edge_mask = self.tmp_edgemask(prob, edge_index)
            _, _, edge_mask = self.forward((x, emb, edge_index, 1.0), training=False)
        return edge_mask

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        # subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
        #     node_idx, self.num_hops, edge_index, relabel_nodes=True,
        #     num_nodes=num_nodes, flow=self.__flow__())

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
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

    def train_NC_explanation_network(self, data):
        # 바꿀 부분

        # dataset_indices = torch.where(
        #     (data.train_mask != 0)
        #     & (torch.isin(data.y, torch.tensor([2, 4, 5, 6]))))[0].tolist()
        # dataset_indices = torch.where((data.train_mask != 0))[0]
        # dataset_indices = torch.where(dataset_indices >= 600)[0].tolist()
        # pass_indices = []


        # dataset_indices = torch.where(data.node_mask)[0].tolist()
        if self.node_indice:
            dataset_indices = self.node_indice
        else:
            dataset_indices = range(300, 700, 5)

        # print(dataset_indices)
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)

        # collect the embedding of nodes
        x_dict = {}
        y_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        emb_dict = {}
        pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            data.to(self.device)
            for gid in tqdm(dataset_indices):
                if (self.dataset.lower() in ['cora', 'citeseer']) and (glob.glob(os.path.join(self.sub_path, f"node_{gid}.pt"))):
                    file = glob.glob(os.path.join(self.sub_path, f"node_{gid}.pt"))[0]
                    x, edge_index, y, subset = torch.load(file, map_location='cpu')
                #
                else:
                    x, edge_index, y, subset, _ = \
                        self.get_subgraph(node_idx=gid, x=data.x,
                                          edge_index=data.edge_index, y=data.y)
                    #
                    if (self.dataset.lower() in ['cora', 'citeseer', 'pubmed']) and (not glob.glob(os.path.join(self.sub_path, f"node_{gid}.pt"))):
                        save_path = os.path.join(self.sub_path, f"node_{gid}.pt")

                        cache_list = [x.cpu(), edge_index.cpu(), y.cpu(), subset.cpu()]
                        torch.save(cache_list, save_path)

                emb, _, prob = self.get_model_output(x, edge_index)

                # if prob[int(torch.where(subset == gid)[0])].argmax(-1) == 0:
                #     pass_indices.append(gid)
                #     continue

                x_dict[gid] = x.cpu()
                y_dict[gid] = y.cpu()
                edge_index_dict[gid] = edge_index.cpu()
                node_idx_dict[gid] = int(torch.where(subset == gid)[0])
                pred_dict[gid] = prob[node_idx_dict[gid]].argmax(-1).cpu()
                emb_dict[gid] = emb.data.cpu()

            data.detach().cpu()
        # train the explanation network

        for epoch in range(self.epochs):
            loss_pge_total = 0.0
            loss_ood_total = 0.0
            loss = 0.0
            max_ent = torch.log(torch.tensor(self.nclass)).item()

            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))

            self.elayers.train()
            for gid in tqdm(dataset_indices):
                # if pred_dict[gid] != data.y[gid]:
                #     continue
                if edge_index_dict[gid].shape[1] == 0:
                    continue

                pred, x3, edge_mask = self.forward(
                    (x_dict[gid], emb_dict[gid], edge_index_dict[gid], tmp)
                    , training=True)
                # exit()
                # print(pred, x3, edge_mask)
                # if pred[node_idx_dict[gid]].sum() != 1.0:
                #     print(gid, pred[node_idx_dict[gid]].sum(), pred[node_idx_dict[gid]])
                # if gid == 300:
                #     print(edge_mask, pred[node_idx_dict[gid]], pred_dict[gid])

                loss_pge = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid])
                loss_ood = self.loss_oodgat(pred, edge_mask, edge_index_dict[gid])

                a = torch.tensor(self.beta).to(self.device)
                b = torch.tensor(0.1).to(self.device)
                if self.beta:
                    loss_tmp = (loss_pge * self.alpha) + (loss_ood * torch.pow(a, b * epoch))
                else:
                    loss_tmp = (loss_pge * self.alpha)

                loss_tmp.backward()
                loss_pge_total += loss_pge.item()
                loss_ood_total += loss_ood.item()
                loss += loss_tmp.item()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss:.4f} | Loss_pge: {loss_pge_total:.4f} * {self.alpha} | Loss_ood: {loss_ood_total:.4f} * {torch.pow(a, b * epoch)})')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)

    def eval_node_probs(self, node_idx: int, x: torch.Tensor,
                        edge_index: torch.Tensor, edge_mask: torch.Tensor, **kwargs):
        probs = self.eval_probs(x=x, edge_index=edge_index,
                                edge_mask=edge_mask, **kwargs)
        return probs[node_idx].squeeze()

    def get_node_prediction(self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        outputs = self.get_model_output(x, edge_index, edge_mask=None, **kwargs)
        return outputs[2][node_idx].argmax(dim=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
