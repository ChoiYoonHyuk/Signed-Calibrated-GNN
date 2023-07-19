import numpy as np
import random
import math
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import Union, Tuple, Optional
from torch_scatter import gather_csr, segment_csr
from sklearn.metrics import f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0.00
lr = 1e-2
epoch = 300
reg = 1
idx = 1

'''np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)'''

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')

data = dataset[0].to(device)
num_class = dataset.num_classes

if dataset.root == '/tmp/Cora' or dataset.root == '/tmp/Citeseer' or dataset.root == '/tmp/Pubmed':
    data.val_mask = torch.logical_not(torch.logical_or(data.train_mask, data.test_mask))


if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Cornell' or dataset.root == '/tmp/Actor' or dataset.root == '/tmp/Texas' or dataset.root == '/tmp/Wisconsin':
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
    #data.train_mask, data.val_mask, data.test_mask = data.train_mask[:, idx], data.val_mask[:, idx], data.test_mask[:, idx]
    
extend_num = 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
edge_lambda = 1.0
temperature = 2.0
loss_func = nn.MSELoss()
l0_loss = nn.L1Loss()
alpha, beta = 1.1, .0

tmp = []
for p in range(len(data.edge_index[0])):
    l_1, l_2 = data.y[data.edge_index[0][p]], data.y[data.edge_index[1][p]]
    if l_1 == l_2:
        tmp.append(1.0)
    else:
        tmp.append(0.0)
edge_weights = torch.tensor(tmp).to(device)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
                    

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
        

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.gcn_1 = GCNConv(dataset.num_node_features, 64)
        self.gcn_2 = GCNConv(64, dataset.num_classes)
        
        self.ptd_1 = nn.Sequential(
            nn.Linear(dataset.num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),)
        self.ptd_2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),)
        
    def forward(self, data, test_idx):
        x, edge_index = data.x, data.edge_index
        
        #eta = torch.zeros(1).to(device).uniform_(0, 1)
        n_cut = sum(edge_weights)
        
        tmp = max(0.05, temperature)
        ones, zeros = torch.ones(1).to(device), torch.zeros(1).to(device)
        gate = torch.FloatTensor(len(data.edge_index[0]), 1).uniform_(1e-6, 1 - 1e-6).to(device)
        gate = torch.log(gate) - torch.log(1 - gate)
        agg = (self.ptd_1(data.x[edge_index[0]] - data.x[edge_index[1]])) * (self.ptd_1(data.x[edge_index[0]] - data.x[edge_index[1]]))
        if test_idx == 0:
            s_1 = alpha * ((agg + gate) / tmp).sigmoid() - beta
        else:
            s_1 = alpha * agg.sigmoid()
        s_1 = s_1.view(-1)
        s_1 = torch.clip(s_1, 0, 1)
        # Normalize
        rowsum = 1 / torch.sqrt(to_dense_adj(edge_index, edge_attr=s_1).squeeze(0).sum(dim=0))
        rowsum = torch.clip(rowsum, 0, 10)
        #s_1 = s_1 * torch.gather(rowsum, 0, edge_index[0]) * torch.gather(rowsum, 0, edge_index[1])
        
        if idx == 0:
            x = self.gcn_1(x, edge_index)
        else:
            x = self.gcn_1(x, edge_index, s_1)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        gate = torch.FloatTensor(len(data.edge_index[0]), 1).uniform_(1e-6, 1 - 1e-6).to(device)
        gate = torch.log(gate) - torch.log(1 - gate)
        #s_2 = alpha * ((self.ptd_2(x[edge_index[0]] + x[edge_index[1]]) + gate) / tmp).sigmoid() - beta
        agg = (self.ptd_2(x[edge_index[0]] - x[edge_index[1]])) * (self.ptd_2(x[edge_index[0]] - x[edge_index[1]]))
        if test_idx == 0:
            s_2 = alpha * ((agg + gate) / tmp).sigmoid() - beta
        else:
            s_2 = alpha * agg.sigmoid()
        s_2 = s_2.view(-1)
        s_2 = torch.clip(s_2, 0, 1)
        rowsum = 1 / torch.sqrt(to_dense_adj(edge_index, edge_attr=s_2).squeeze(0).sum(dim=0))
        rowsum = torch.clip(rowsum, 0, 10)
        #s_2 = s_2 * torch.gather(rowsum, 0, edge_index[0]) * torch.gather(rowsum, 0, edge_index[1])
        #thresh = torch.topk(s_2, n_cut)[0][n_cut-1]
        
        #print(s_1, s_2)
    
        if idx == 0:
            x_out = self.gcn_2(x, edge_index)
        else:
            x_out = self.gcn_2(x, edge_index, s_2)
        
        consist_label = torch.zeros(len(edge_index[0])).to(device)
        consist_loss = loss_func(s_1 - torch.mean(s_1), consist_label) + loss_func(s_2 - torch.mean(s_2), consist_label)
        #consist_loss = loss_func(s_1, torch.mean(s_1)) + loss_func(s_2, torch.mean(s_2))
        
        s_1_dense, s_2_dense = to_dense_adj(data.edge_index, edge_attr=s_1).squeeze(0) + 1e-6, to_dense_adj(data.edge_index, edge_attr=s_2).squeeze(0) + 1e-6
        
        aa_1, aa_2 = torch.matmul(s_1_dense, s_1_dense.T), torch.matmul(s_2_dense, s_2_dense.T)
        #adj_1, adj_2 = to_dense_adj(data.edge_index).squeeze(0), to_dense_adj(data.edge_index).squeeze(0)
        
        U1, s1, V1 = torch.svd_lowrank(s_1_dense)
        U2, s2, V2 = torch.svd_lowrank(s_2_dense)
        
        '''U1.requires_grad = False
        s1.requires_grad = False
        V1.requires_grad = False
        U2.requires_grad = False
        s2.requires_grad = False
        V2.requires_grad = False'''
        
        nuc_loss = 0
        
        for k in range(len(V1[0])):
            vi = V1[:, k].view(-1, 1)
            vi = torch.matmul(aa_1, vi)
            vi_norm = torch.linalg.norm(vi)
            vi = vi / vi_norm
            vmv = torch.matmul(vi.T, torch.matmul(aa_1, vi))
            vv = torch.matmul(vi.T, vi) #+ 1e-4
            
            nuc_loss += torch.sqrt(torch.abs(vmv / vv))
                        
            if k > 1:
                aa_minus = torch.mul(aa_1, torch.mul(vi, vi.T))
                aa_1 = aa_1 - aa_minus
        
        for k in range(len(V2[0])):
            vi = V2[:, k].view(-1, 1)
            vi = torch.matmul(aa_2, vi)
            vi_norm = torch.linalg.norm(vi)
            vi = vi / vi_norm
            
            vmv = torch.matmul(vi.T, torch.matmul(aa_2, vi))
            vv = torch.matmul(vi.T, vi) #+ 1e-4
            
            nuc_loss += torch.sqrt(torch.abs(vmv / vv))
                        
            if k > 1:
                aa_minus = torch.matmul(aa_2, torch.matmul(vi, vi.T))
                aa_2 = aa_2 - aa_minus
                
        z_loss = torch.abs(torch.sum(edge_weights) - torch.sum(s_1)) + torch.abs(torch.sum(edge_weights) - torch.sum(s_2))
        
        calib = - torch.max(x_out, 1)[0] + torch.topk(x_out, 2)[0][:, 1]
                
        return F.log_softmax(x_out, dim=1), F.softmax(x_out, dim=1), 0.01 * consist_loss + 0.01 * nuc_loss, s_1, s_2, torch.mean(calib)
        

gnn = GNN().to(device)

gnn.train()

gnn_optim = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=5e-4)

epoch, early_stop, best_epoch = 300, 0, 0
agg_thresh = 0.5
best = 100.0

best_gnn = 0
best_valid = 0
classification = 0
early_stop = 0
train_loss = 100
edge_weight = 0
best_neg_85, best_neg_99 = 0, 0
best_mse = 10
#data.valid_mask = torch.logical_not(torch.logical_or(data.train_mask, data.test_mask))

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
for e in range(100):    
    temperature = temperature * 0.99
    
    for _ in tqdm(range(200)):                  
        gnn.train()
        
        out, _, inte_loss, _, _, calib = gnn(data, 0)
        
        gnn_optim.zero_grad()
        if idx == 0:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + calib * .01
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + inte_loss + calib * .01
        loss.backward()
        gnn_optim.step()
        
        with torch.no_grad():
            gnn.eval()
            pred, x, _, s1, s2, _ = gnn(data, 1)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
            '''n_cut = int(sum(edge_weights))
            err = 0
            p_cor, n_cor, nn_cor, nnn_cor = 0, 0, 0, 0
            thresh = torch.topk(s1, n_cut)[0][n_cut-1]
            s2_thresh = torch.topk(s2, n_cut)[0][n_cut-1]
            thresh_95 = torch.topk(s1, int(len(data.edge_index[0]) * 0.95))[0][int(len(data.edge_index[0]) * 0.95)-1]
            thresh_99 = torch.topk(s1, int(len(data.edge_index[0]) * 0.99))[0][int(len(data.edge_index[0]) * 0.99)-1]
            s2_ew = torch.where(s2 > s2_thresh, torch.ones(1).to(device), torch.zeros(1).to(device))'''
            
            if valid > best_valid:
                best_valid = valid
                best_gnn = acc
                print(best_gnn)
                save = x
                
                '''x1, x2 = x[data.edge_index[0]], x[data.edge_index[1]]
                out = torch.sum(x1*x2, dim=1)
                thresh = torch.topk(out, int(len(data.edge_index[0]) * 0.81))[0][int(len(data.edge_index[0]) * 0.81) - 1]
                e_w = torch.where(out > thresh, torch.ones(1).to(device), torch.zeros(1).to(device))
                mat = torch.sum(torch.logical_not(torch.logical_xor(edge_weights, e_w)))
                print('PTDNet %.2f' % float(mat / len(data.edge_index[0])))
                print('Recall %.4f' % f1_score(s2_ew.detach().cpu().numpy(), edge_weights.detach().cpu().numpy()))
                
                for idx in range(len(data.edge_index[0])):  
                    e_1, e_2 = data.edge_index[0][idx], data.edge_index[1][idx]
                    l_1, l_2 = data.y[e_1], data.y[e_2]
                    if int(l_1) == int(l_2):
                        err += abs(1 - float(s1[idx]))
                    else:
                        err += abs(float(s1[idx]))
                        if s1[idx] < thresh:
                            n_cor += 1
                        if s1[idx] < thresh_95:
                            nn_cor += 1
                        if s1[idx] < thresh_99:
                            nnn_cor += 1
                
                if n_cor / int(len(data.edge_index[0]) - n_cut) > best_neg_85:
                    best_neg_85 = n_cor / (len(data.edge_index[0]) - n_cut)
                if nnn_cor / int(len(data.edge_index[0]) * 0.01) > best_neg_99:
                    best_neg_99 = nnn_cor / int(len(data.edge_index[0]) * 0.01)
                if err / len(data.edge_index[0]) < best_mse:
                    best_mse = err / len(data.edge_index[0])'''
                #print(min(s1), torch.mean(s1), torch.max(s1))
                #print(err / len(data.edge_index[0]), n_cor / (len(data.edge_index[0])-n_cut), nn_cor / int(len(data.edge_index[0]) * 0.05), nnn_cor / int(len(data.edge_index[0]) * 0.01))
                #print(best_mse, best_neg_85, best_neg_99)
    if e == 1:
        print(dissonance(save))
        exit()