import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
from tqdm import tqdm
import argparse

from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import torch.nn.init as init
from torch_geometric.nn import inits
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

use_calib = 0
edge_calib = 0

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    thresh = .9
    lam = .1
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    lam = .1
    thresh = .9
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    lam = .1
    thresh = .9
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    lam = .001
    thresh = .3
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    lam = .001
    thresh = 0
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    lam = .1
    thresh = .5
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    lam = .001
    thresh = .5
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    lam = .001
    thresh = .5
elif data_id == 8:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    lam = .001
    thresh = .5

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

class GGCNlayer_SP(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer_SP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            self.adj_remove_diag = None
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)
    
    def precompute_adj_wo_diag(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_wo_diag_ind = (adj_i[0,:]!=adj_i[1,:])
        self.adj_remove_diag = torch.sparse.FloatTensor(adj_i[:,adj_wo_diag_ind], adj_v[adj_wo_diag_ind], adj.size())
                        
    def non_linear_degree(self, a, b, s):
        i = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(i, self.sftpls(a*v+b), s.size())
    
    def get_sparse_att(self, adj, Wh):
        i = adj._indices()
        Wh_1 = Wh[i[0,:],:]
        Wh_2 = Wh[i[1,:],:]
        sim_vec = F.cosine_similarity(Wh_1, Wh_2)
        sim_vec_pos = F.relu(sim_vec)
        sim_vec_neg = -F.relu(-sim_vec)
        return torch.sparse.FloatTensor(i, sim_vec_pos, adj.size()), torch.sparse.FloatTensor(i, sim_vec_neg, adj.size())
    
    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.non_linear_degree(self.deg_coeff[0], self.deg_coeff[1], degree_precompute)

        Wh = self.fcn(h)
        if self.use_sign:
            if self.adj_remove_diag is None:
                self.precompute_adj_wo_diag(adj)
        if self.use_sign:
            e_pos, e_neg = self.get_sparse_att(adj, Wh)
            if self.use_degree:
                attention_pos = self.adj_remove_diag*sc*e_pos
                attention_neg = self.adj_remove_diag*sc*e_neg
            else:
                attention_pos = self.adj_remove_diag*e_pos
                attention_neg = self.adj_remove_diag*e_neg
            
            prop_pos = torch.sparse.mm(attention_pos, Wh)
            prop_neg = torch.sparse.mm(attention_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.sparse.mm(adj*sc, Wh)
            else:
                prop = torch.sparse.mm(adj, Wh)
            
            result = prop
        return result

class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)


    
    def forward(self, h, adj, degree_precompute, test_idx):
        if self.use_degree:
            sc = self.deg_coeff[0]*degree_precompute+self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod),1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod/torch.max(torch.sqrt(scaling),1e-9*torch.ones_like(scaling))
            e = e-torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e*adj*sc
            else:
                attention = e*adj
            
            #tmp = torch.where((sim > 0) & (minus_attr < .0), (1 - sim) * minus_attr, minus_attr)
            #tmp = torch.where((sim < thresh) & (minus_attr < .0), torch.zeros(1).to(device), minus_attr)
            
            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)
            if test_idx:
                norm = Wh / Wh.norm(dim=1)[:, None]
                sim = torch.mm(norm, norm.T) * torch.where(adj > 0, torch.ones(1).to(device), torch.zeros(1).to(device))
                calib_neg = torch.where((attention_neg < 0) & (sim > thresh), torch.zeros(1).to(device), attention_neg)
                prop_neg = torch.matmul(calib_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)
            
        else:
            if self.use_degree:
                prop = torch.matmul(adj*sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)
            
            result = prop
                 
        return result

        
        
class GGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, decay_rate, exponent, use_degree=True, use_sign=True, use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False):
        super(GGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if use_sparse:
            model_sel = GGCNlayer_SP
        else:
            model_sel = GGCNlayer
        self.convs.append(model_sel(nfeat, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        for _ in range(nlayers-2):
            self.convs.append(model_sel(nhidden, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.convs.append(model_sel(nhidden, nclass, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.fcn = nn.Linear(nfeat, nhidden)
        self.act_fn = F.elu
        self.dropout = dropout
        self.use_decay = use_decay
        if self.use_decay:
            self.decay = decay_rate
            self.exponent = exponent
        self.degree_precompute = None
        self.use_degree = use_degree
        self.use_sparse = use_sparse
        self.use_norm = use_bn or use_ln
        if self.use_norm:
            self.norms = nn.ModuleList()
        if use_bn:
            for _ in range(nlayers-1):
                self.norms.append(nn.BatchNorm1d(nhidden))
        if use_ln:
            for _ in range(nlayers-1):
                self.norms.append(nn.LayerNorm(nhidden))
    
    def precompute_degree_d(self, adj):
        diag_adj = torch.diag(adj)
        diag_adj = torch.unsqueeze(diag_adj, dim=1)
        self.degree_precompute = diag_adj/torch.max(adj, 1e-9*torch.ones_like(adj))-1
    
    def precompute_degree_s(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_diag_ind = (adj_i[0,:]==adj_i[1,:])
        adj_diag = adj_v[adj_diag_ind]
        v_new = torch.zeros_like(adj_v)
        for i in range(adj_i.shape[1]):
            v_new[i] = adj_diag[adj_i[0,i]]/adj_v[i]-1
        self.degree_precompute = torch.sparse.FloatTensor(adj_i, v_new, adj.size())
    
    
    def forward(self, x, adj, test_idx):
        if self.use_degree:
            if self.degree_precompute is None:
                if self.use_sparse:
                    self.precompute_degree_s(adj)
                else:
                    self.precompute_degree_d(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = self.fcn(x)
        layer_previous = self.act_fn(layer_previous)
        layer_inner = self.convs[0](x, adj, self.degree_precompute, test_idx)

        for i,con in enumerate(self.convs[1:]):
            if self.use_norm:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = self.act_fn(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if i==0:
                layer_previous = layer_inner + layer_previous
            else:
                if self.use_decay:
                    coeff = math.log(self.decay/(i+2)**self.exponent+1)
                else:
                    coeff = 1
                layer_previous = coeff*layer_inner + layer_previous
            layer_inner = con(layer_previous,adj,self.degree_precompute,test_idx)
        
        pred = F.softmax(layer_inner, dim=1)
        calib_len = len(data.x)
        calib = torch.max(pred, 1)[0] + torch.topk(pred, 2)[0][:, 1]
        
        return F.log_softmax(layer_inner, dim=1), torch.mean(calib), F.softmax(layer_inner, dim=1)


model = GGCN(len(data.x[0]), 2, 64, dataset.num_classes, 0.5, 1.0, 3.0).to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
epoch, best_value, best_epoch = 500, 0, 0
best_valid = 0


edge_weight = torch.ones(data.edge_index.size(1)).to(device)
edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
adj = to_dense_adj(edge_index).squeeze(0)

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
        
        out, calib, _ = model(data.x, adj, 0)
    
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + calib * lam * use_calib
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, _, xs = model(data.x, adj, 1)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                        
            if valid > best_valid:
                best_valid = valid
                best_value = acc
                print('%.3f' % best_value)
                save = xs
    