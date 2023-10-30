import torch
import argparse

import pickle
import numpy as np
from gcn_syn2 import GCN

from tqdm import tqdm
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BA_Community')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')
parser.add_argument('--numbering', default=0)
parser.add_argument('--mode', default='test')
parser.add_argument('--version', default='v3')
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
seed = 66
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

nclass = 4
with open(f'../dataset/{args.dataset}/{args.dataset}.pkl', 'rb') as fin:
    data = pickle.load(fin)


if 'community' in args.dataset.lower():
    nclass = 8


# GPU 설정
data = data.to(device)
model = GCN(nfeat=data.x.shape[1], nhid=args.hidden, nclass=nclass, device=device, dropout=args.dropout)

train_mask = data.train_mask
test_mask = data.test_mask

if args.mode == 'train':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    # # 훈련


    model = model.to(device)

    for epoch in tqdm(range(args.epochs)):
        # data.y = data.y.type(torch.LongTensor)
        model.train()  # 학습 준비
        _, logits, out = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])# 손실함수 계산

        # print(out[data.train_mask])
        optimizer.zero_grad()  # 파라미터 초기화
        loss.backward()# 역전파
        optimizer.step()# 파라미터 업데이터

        if epoch%200==0:
            #모든 값이 0이 되어버림...
            a, pred = out.max(dim=1)
            # print(pred[test_mask])
            acc = (data.y[data.test_mask] == pred[data.test_mask]).sum() / data.test_mask.sum()
            print("{} acc: {} loss:{} {} {}".format(epoch+1, acc, loss, (data.y[data.test_mask] == pred[data.test_mask]).sum(), data.test_mask.sum()))

    _, logits, out = model(data)
    _, pred = out.max(dim=1)
    acc = (data.y[data.test_mask] == pred[data.test_mask]).sum() / data.test_mask.sum()
    print("{} acc: {} loss:{} {} {}".format(epoch+1, acc, loss, (data.y[data.test_mask] == pred[data.test_mask]).sum(),
                                            data.test_mask.sum()))

    model.cpu()
    model.eval()
    torch.save(model.state_dict(), "./models/gcn_{}_epochs{}_lr{}_{}".format(args.dataset, args.epochs, args.lr, args.version) + ".pt")

# 옵티마이저 생성
# 모델 평가
model.load_state_dict(torch.load("./models/gcn_{}_epochs{}_lr{}_{}".format(args.dataset, args.epochs, args.lr, args.version) + ".pt"))
model.to(device)

model.eval()
x3, logits, prob = model(data)

_, pred = prob.max(dim=1)
# 테스트

# acc = int(data == pred).sum() ) / len(test_indices)
# print(data.node_mask)
acc = (data.y[data.node_mask.bool()] == pred[data.node_mask.bool()]).sum() / len(data.y[data.node_mask.bool()])
print("{} acc: {} {} {}".format('final', acc,  (data.y[data.node_mask.bool()] == pred[data.node_mask.bool()]).sum(), len(data.y[data.node_mask.bool()])))


for i in range(8):
    print(f'{i}: {(pred==i).sum()}\t'
          f'correct {len(torch.where((pred==i) & (data.y==i))[0])}')

# exit()

sizes = [320]
versions = ['v4', 'v5', 'v6', 'v7']

for size in sizes:
    for version in versions:
        with open(f'../dataset/{args.dataset}/{args.dataset}_{size}_{version}.pkl', 'rb') as fin:
            data = pickle.load(fin)
            data.to(device)
        model.eval()
        print(f'size: {size} version: {version}')
        print(data)
        out, pred = model(data)[2].max(dim=1)
        # 테스트
        # print(F.cross_entropy(torch.exp(model(data)[1][:size]), torch.exp(model(data)[1][:size]), reduction='none'))
        # acc = int(data == pred).sum() ) / len(test_indices)
        acc = (data.y[:1400] == pred[:1400]).sum() / len(data.y[:1400])
        a = 0
        b = 0
        c = 0
        print("{} acc: {} {} {}".format('final', acc,  (data.y[:1400] == pred[:1400]).sum(), len(data.y[:1400])))
        for i in range(8):
            a += (pred[data.node_mask]==i).sum()
            b += len(torch.where((pred[data.node_mask]==i) & (data.y[data.node_mask]==i))[0])
            print(f'c: {(pred[1400:]==i).sum()}')
            print(f'{i}: {(pred[data.node_mask]==i).sum()}\t'
                  f'correct {len(torch.where((pred[data.node_mask]==i) & (data.y[data.node_mask]==i))[0])}')

        prob = model(data)[2]

        print(out[range(0, 1400)].mean().item())
        print(out[range(1400, 1720)].mean().item())
        print(f"b: {b}, a: {a}, b/a: {b/a}\n\n")

        print((pred[data.node_mask]==data.y[data.node_mask]).sum())
