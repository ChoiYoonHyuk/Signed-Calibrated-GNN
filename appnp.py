import numpy as np
import random
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Linear(dataset.num_node_features, 64)
        self.appnp = APPNP(10, .1)
        self.pred = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, dataset.num_classes),
        )
        self.pred = nn.Linear(64, dataset.num_classes)
        
    def forward(self, data, e_w, idx):
        x, edge_index = data.x, data.edge_index
        err = 0
        
        x = F.dropout(F.relu(self.embed(x)))
        x = self.appnp(x, edge_index)
        x_out = self.pred(x)
            
        return F.log_softmax(x_out, dim=1), x_out, err, F.softmax(x_out, dim=1)


#data.x = data.x * 1.05 - 0.05

out = 0
tmp = []
best_value = 0
idx = 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
param = 1.0


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
        
    for _ in range(300):    
        model.train()
        
        out, _, err, _ = model(data, data.edge_index, 0)
    
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) * reg #+ err
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, x, _, xs = model(data, data.edge_index, 0)
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
