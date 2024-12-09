import numpy as np
import random
import math
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0.00
lr = 1e-3
reg = 1.0

#np.random.seed(seed)
#random.seed(seed)
#torch.manual_seed(seed)

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

data = dataset[0].to(device)

num_class = dataset.num_classes

idx = 0
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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn_1 = GCNConv(dataset.num_node_features, 64)
        
        self.gcn_2 = GCNConv(64, num_class)
        
    def forward(self, data, e_w, idx):
        x, edge_index = data.x, data.edge_index
        
        if idx == 0:
            x = F.dropout(F.relu(self.gcn_1(x, edge_index)))
            x_out = self.gcn_2(x, edge_index)
            
            cosine_sim = 1 - F.cosine_similarity(x_out[edge_index[0]], x_out[edge_index[1]])
            gnn_edge = torch.where(edge_weights > 0.5, torch.ones(1).to(device), torch.empty((1)).fill_(-1.0).to(device))
            #err = torch.mean(gnn_edge * cosine_sim * l_u_edges)
            err = 0
            
        else:
            x = F.dropout(F.relu(self.gcn_1(x, edge_index, e_w)))
            x_out = self.gcn_2(x, edge_index, e_w)        
            err = 0
            
        return F.log_softmax(x_out, dim=1), x_out, err, F.softmax(x_out, dim=1)


#data.x = data.x * 1.05 - 0.05

out = 0
tmp = []
best_value = 0
idx = 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
param = 1.0

for p in range(len(data.edge_index[0])):
    e_1, e_2 = data.edge_index[0][idx], data.edge_index[1][idx]  
    if data.train_mask[e_1] == False and data.train_mask[e_2] == False:
        l_u_edges[p] = .5
    if data.train_mask[e_1] == False or data.train_mask[e_2] == False:
        l_u_edges[p] = 1.0
        
    l_1, l_2 = data.y[data.edge_index[0][p]], data.y[data.edge_index[1][p]]
    k = random.randrange(0, 100)
    if l_1 == l_2:
        if k < idx:
            tmp.append(param)
        else:
            tmp.append(0.0)
    else:
        if k < idx:
            tmp.append(0.0)
        else:
            tmp.append(1.0)
        
edge_weights = torch.tensor(tmp).to(device)

model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, early_stop, best_epoch, idx = 500, 0, 0, 0
best_valid = 0
e_w = 0

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

tmp2, best_x = [], 0
save = 0
for ee in tqdm(range(epoch)):
    #if ee > 0:
        #model.load_state_dict(torch.load('./param/gcn_' + dataset.name + '.pth'))
        
    for _ in range(1000):    
        model.train()
        
        out, _, err, _ = model(data, edge_weights, 0)
    
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) * reg #+ err
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, x, _, xs = model(data, edge_weights, 0)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
            if valid > best_valid:
                best_valid = valid
                best_value, best_epoch = acc, early_stop
                best_x = x
                save = xs
    if ee > 10:
        print(dissonance(save))
        exit()
    print(best_valid, best_value)
