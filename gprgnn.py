import numpy as np
import random
import math
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import torch.nn.init as init
from torch_geometric.nn import inits
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
#from torch_geometric.nn import FAConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from sklearn.metrics import f1_score

from typing import Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

use_calib = 1

lr = 1e-3
parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    alpha, sign = .1, 1
    lam = .01
    thresh = .9
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    alpha, sign = .1, 1
    lam = .01
    thresh = .9
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    alpha, sign = .1, 1
    lam = .01
    thresh = .9
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    alpha, sign = .9, -1
    lam = .001
    thresh = .9
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    alpha, sign = .9, -1
    lam = .05
    thresh = -.9
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    alpha, sign = .9, 1
    lam = .01
    thresh = .8
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    alpha, sign = .9, -1
    lam = .001
    thresh = .9
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    alpha, sign = .9, -1
    lam = .001
    thresh = .9
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    alpha, sign = .9, -1
    lam = .001
    thresh = .9

data = dataset[0].to(device)
num_class = dataset.num_classes

idx = 0
if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Actor':
    num_class = 5
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(5):
        train_mask.extend(labels[c][0:20])
        cut = int((len(labels[c]) - 20) / 2)
        valid_mask.extend(labels[c][20:20+cut])
        test_mask.extend(labels[c][20+cut:len(labels[c])])
    
    train, valid, test = [], [], []
    for x in range(len(data.y)):
        if x in train_mask:
            train.append(True)
            valid.append(False)
            test.append(False)
        elif x in valid_mask:
            train.append(False)
            valid.append(True)
            test.append(False)
        elif x in test_mask:
            train.append(False)
            valid.append(False)
            test.append(True)
        else:
            train.append(False)
            valid.append(False)
            test.append(True)
    
    data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
    data.y = torch.where(data.y > 4, 0, data.y)

if dataset.root == '/tmp/Cornell' or dataset.root == '/tmp/Texas' or dataset.root == '/tmp/Wisconsin':
    num_class = 5
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(5):
        train_mask.extend(labels[c][0:5])
        cut = int((len(labels[c]) - 5) / 2)
        valid_mask.extend(labels[c][5:5+cut])
        test_mask.extend(labels[c][5+cut:len(labels[c])])
    
    train, valid, test = [], [], []
    for x in range(len(data.y)):
        if x in train_mask:
            train.append(True)
            valid.append(False)
            test.append(False)
        elif x in valid_mask:
            train.append(False)
            valid.append(True)
            test.append(False)
        elif x in test_mask:
            train.append(False)
            valid.append(False)
            test.append(True)
        else:
            train.append(False)
            valid.append(False)
            test.append(True)
    
    data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
    data.y = torch.where(data.y > 4, 0, data.y)

edge_weight = torch.ones(data.edge_index.size(1)).to(device)
edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
num_nodes = len(data.x)
row, col = edge_index[0], edge_index[1]
deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
deg_inv_sqrt = deg.pow_(-0.5)
deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
sym = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dataset.num_node_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, dataset.num_classes),
        )
        self.K = 10
        TEMP = sign*alpha*(1-alpha)**np.arange(self.K+1)
        TEMP[-1] = sign*(1-alpha)**self.K
        self.temp = Parameter(torch.tensor(TEMP))
        self.alpha = alpha
        #self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = sign * self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = sign * (1-self.alpha)**self.K
        
    def forward(self, data, e_w, test_idx):
        x, edge_index = data.x, data.edge_index
        
        x = self.embed(x)
        out = x * self.temp[0]
                
        for k in range(self.K):
            tmp = sym * self.temp[k+1]
            norm = x / (x.norm(dim=1) + 1e-6)[:, None]
            sim = torch.mm(norm, norm.T)
            outs = torch.where((tmp < 0) & (sim > thresh), torch.zeros(1).to(device), sym)
            
            if test_idx:
                x = torch.mm(outs, x)
            else:
                x = torch.mm(sym, x)
            out += self.temp[k+1] * x
        
        pred = F.softmax(out, dim=1)
        calib_len = len(data.x)
        calib = 1 - torch.max(out, 1)[0] + torch.topk(out, 2)[0][:, 1]
            
        return F.log_softmax(out, dim=1), torch.mean(calib), F.softmax(out, dim=1)


model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, best_value, best_epoch = 500, 0, 0
best_valid = 0


def bal(x, idx):
    res = 0
    for i in range(num_class):
        if i != idx:
            if x[i]+x[idx] < 1e-3:
                res += x[i] * (1 - abs(x[i]-x[idx])/1e-3)
            else:
                res += x[i] * (1 - abs(x[i]-x[idx])/(x[i]+x[idx]))
    return res


def dissonance(x):
    dis = 0
    for i in range(len(data.x)):
        for j in range(num_class):
            if sum(x[i]) - x[i][j] < 1e-3:
                dis += (x[i][j] * bal(x[i], j)) / 1e-3
            else:
                dis += (x[i][j] * bal(x[i], j)) / (sum(x[i]) - x[i][j])
    return dis / len(data.x)
    
save = 0
for ite in tqdm(range(epoch)):
    for _ in range(1000):    
        model.train()
        
        out, calib, _ = model(data, data.edge_index, 0)
    
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + calib * lam * use_calib
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, _, xs = model(data, data.edge_index, 1)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                        
            if valid > best_valid:
                best_valid = valid
                best_value = acc
                print('%.3f' % best_value)
                save = xs
    
