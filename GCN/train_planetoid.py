import torch
import pickle
import argparse
from tqdm import tqdm

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from gcn_cite import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Citeseer')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--version', default='v3')
args = parser.parse_args()
# device = 'cpu'

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

with open(f'../dataset/{args.dataset}/{args.dataset}.pkl'.format(args.dataset), 'rb') as fin:
    data = pickle.load(fin)

nfeat = data.x.shape[1]

nclass = len(data.id_class.unique()) - 1

train_mask = data.train_mask.bool()
valid_mask = data.valid_mask.bool()
test_mask = data.test_mask.bool()


id_data = Data(x=data.x, y=data.id_class, edge_index=data.id_edge_index)
model = GCN(nfeat=nfeat, nhid=args.hidden,
            nclass=nclass, dropout=args.dropout, device=device).to(device)
epoch=0

id_data.to(device)
data.to(device)

if args.mode=='train':

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()



    model.train()
    for epoch in tqdm(range(args.epochs)):

        optimizer.zero_grad()  # 파라미터 초기화
        _, logits, out = model(id_data)
        # print(out[train_mask])
        loss = criterion(out[train_mask], id_data.y[train_mask])  # 손실함수 계산
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이터

        if epoch % 200 == 0:
            # 모든 값이 0이 되어버림...
            a, pred = out.max(dim=1)
            # print(pred[test_mask])
            acc = (id_data.y[valid_mask] == pred[valid_mask]).sum() / len(data.y[valid_mask])
            # print(pred[test_mask], data.y[test_mask])
            print("val {} acc: {} loss:{} {} {}".format(epoch, acc, loss, (id_data.y[valid_mask] == pred[valid_mask]).sum(),
                                                    len(id_data.y[valid_mask])))

    model.train()
    _, logits, out = model(id_data)
    a, pred = out.max(dim=1)
    # print(pred[test_mask])
    acc = (id_data.y[valid_mask] == pred[valid_mask]).sum() / len(id_data.y[valid_mask])
    # print(pred[test_mask], data.y[test_mask])
    print("test {} acc: {} loss:{} {} {}".format(epoch, acc, loss, (id_data.y[valid_mask] == pred[valid_mask]).sum(),
                                            len(id_data.y[valid_mask])))

    model.cpu()
    torch.save(model.state_dict(), "./models/gcn_{}_epochs{}_lr{}_{}".format(args.dataset, args.epochs, args.lr, args.version) + ".pt")

# 모델 평가
model.load_state_dict(torch.load(f"./models/gcn_{args.dataset}_epochs{args.epochs}_lr{args.lr}_{args.version}.pt"))
model.to(device)
model.eval()

_, logits, out = model(id_data)
a, pred = out.max(dim=1)
# print(pred[test_mask])
acc = (id_data.y[test_mask] == pred[test_mask]).sum() / len(id_data.y[test_mask])
# print(pred[test_mask], data.y[test_mask])
print("test_id {} acc: {} {} {}".format(epoch, acc, (id_data.y[test_mask] == pred[test_mask]).sum(),
                                        len(id_data.y[test_mask])))

_, logits, out = model(data)
a, pred = out.max(dim=1)
# print(pred[test_mask])
acc = (id_data.y[test_mask] == pred[test_mask]).sum() / len(id_data.y[test_mask])
# print(pred[test_mask], data.y[test_mask])
print("test_full {} acc: {} {} {}".format(epoch, acc, (id_data.y[test_mask] == pred[test_mask]).sum(),
                                        len(id_data.y[test_mask])))

print(a[test_mask | train_mask | valid_mask].mean())
print(a[~(test_mask | train_mask | valid_mask)].mean())
