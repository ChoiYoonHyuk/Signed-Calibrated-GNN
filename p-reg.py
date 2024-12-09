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
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

seed = 0
itera = 0
move = 0.00
lr = 1e-2
reg = 1.0
mu = .01


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
    

mse_loss = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()
num_class = dataset.num_classes

out = 0
idx = 0
tmp = []
best_value = 0
idx_1, idx_2 = 100, 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
param = 1.0
jj = 0


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
    data.y = torch.where(data.y > 4, 0, data.y)
    #data.train_mask, data.val_mask, data.test_mask = data.train_mask[:, idx], data.val_mask[:, idx], data.test_mask[:, idx]

            
edge_weights = torch.tensor(tmp).to(device)

adj = to_dense_adj(data.edge_index).squeeze(0)
#adj_opt = to_dense_adj(data.edge_index, edge_attr=edge_weights).squeeze(0)
deg = torch.sum(adj, dim=0).view(-1, 1)
deg = 1 / torch.where(deg < 1, torch.empty(1).fill_(10^6).to(device), deg)


minus, zero = [], []
k = 100
for p in range(len(data.edge_index[0])):
    l_1, l_2 = data.y[data.edge_index[0][p]], data.y[data.edge_index[1][p]]
    ran = random.randrange(0,100)
    
    if l_1 == l_2:
        if ran < k:
            minus.append(1.0)
            zero.append(1.0)
        else:
            minus.append(-1.0)
            zero.append(0.0)
    else:
        if ran < k:
            minus.append(-1.0)
            zero.append(0.0)
        else:
            minus.append(1.0)
            zero.append(1.0)
minus, zero = torch.tensor(minus).to(device), torch.tensor(zero).to(device)

minus, zero = add_self_loops(data.edge_index, minus), add_self_loops(data.edge_index, zero)
minus_edge, minus_attr = minus[0], minus[1]
zero_edge, zero_attr = zero[0], zero[1]

opt_adj = to_dense_adj(zero_edge, edge_attr=zero_attr).squeeze(0)
opt_adj = to_dense_adj(minus_edge, edge_attr=minus_attr).squeeze(0)


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
            
            _, label = x_out.max(dim=1)
            #_, label = torch.mm(adj, x_out).max(dim=1)
            
            #preg = cross_entropy(torch.mm(adj, x_out) * deg, label)
            #preg = cross_entropy(torch.mm(opt_adj, x_out) * deg, x_out)
            
            preg = cross_entropy(torch.mm(adj, x_out) * deg, data.y) 
            #preg = cross_entropy(torch.mm(opt_adj, x_out) * deg, data.y)
            
            #preg = mse_loss(x_out, torch.mm(opt_adj, x_out) * deg)
            
        else:
            x = F.dropout(F.relu(self.gcn_1(x, edge_index, e_w)))
            x_out = self.gcn_2(x, edge_index, e_w)        
            preg = 0
            #print(F.log_softmax(x_out, dim=1))
        
        return F.log_softmax(x_out, dim=1), x_out, preg


#data.x = data.x * 1.05 - 0.05

model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, early_stop, best_epoch, idx = 500, 0, 0, 0
best_valid = 0
e_w = 0

tmp2, best_x = [], 0

for ee in tqdm(range(epoch)):
    #if ee > 0:
        #model.load_state_dict(torch.load('./param/gcn_' + dataset.name + '.pth'))
        
    for _ in range(300):    
        model.train()
        
        out, _, preg = model(data, edge_weights, 0)
    
        optim.zero_grad()
        itera += 1
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) * reg + preg * mu #math.pow(mu, itera)
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, x, _ = model(data, edge_weights, 0)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
            if valid > best_valid:
                best_valid = valid
                best_value, best_epoch = acc, early_stop
                best_x = x
                k, y = F.softmax(best_x, dim=1).max(dim=1)
    print(best_value)

