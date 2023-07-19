import numpy as np
import random
import math
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import torch.nn.init as init
from torch_geometric.nn import inits
from torch_geometric.utils import add_self_loops, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
import torch.nn.functional as F
#from torch_geometric.nn import FAConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from sklearn.metrics import f1_score

from typing import Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


use_calib = 0
use_thresh = 0

lr = 1e-3
parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    eps = .2
    layers = 4
    lam = .1
    thresh = .9
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    eps = .2
    layers = 4
    lam = .01
    thresh = .8
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    eps = .3
    layers = 4
    lam = .01
    thresh = .9
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    eps = .4
    layers = 2
    lam = .001
    thresh = .9
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    eps = .3
    layers = 2
    lam = .1
    thresh = 0
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    eps = .5
    layers = 2
    lam = .005
    thresh = .8
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    eps = .5
    layers = 2
    lam = .005
    thresh = .8
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    eps = .5
    layers = 2
    lam = .005
    thresh = .8
elif data_id == 8:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    eps = .5
    layers = 2
    lam = .005
    thresh = .8

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

class Linear(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")


    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, torch.nn.parameter.UninitializedParameter):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, '_hook')


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(abs(edge_weight), idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class FAConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(self, channels: int, eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(FAConv, self).__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        self.att_l = Linear(channels, 1, bias=False)
        self.att_r = Linear(channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj, test_idx, 
                edge_weight: OptTensor = None, return_attention_weights=None):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index, Tensor):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)
        
        # propagate_type: (x: Tensor, alpha: PairTensor, edge_weight: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), edge_weight=edge_weight, test_idx=test_idx, size=None)
        
        alpha = self._alpha
        self._alpha = None

        if self.eps != 0.0:
            out = out + self.eps * x_0

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, edge_index, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, edge_weight: OptTensor, test_idx) -> Tensor:
        assert edge_weight is not None
        alpha = (alpha_j + alpha_i).tanh().squeeze(-1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        sim = F.cosine_similarity(x_j[edge_index[0]], x_j[edge_index[1]])
        
        if test_idx & use_thresh:
            alpha = torch.where((sim > thresh) & (alpha < .0), torch.zeros(1).to(device), alpha)
        
        return x_j * (alpha * edge_weight).view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, eps={self.eps})'




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Linear(dataset.num_node_features, 64)
        self.fagcn = FAConv(64, eps=eps, dropout=.5)
        self.pred = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, dataset.num_classes),
        )
        self.pred = nn.Linear(64, dataset.num_classes)
        
    def forward(self, data, e_w, idx, test_idx):
        x, edge_index = data.x, data.edge_index
        
        x_init = F.dropout(F.relu(self.embed(x)))
        
        x = self.fagcn(x_init, x_init, edge_index, test_idx)
        if layers == 2:
            x = self.fagcn(x, x_init, edge_index, test_idx)
        else:
            x = self.fagcn(x, x_init, edge_index, test_idx)
            x = self.fagcn(x, x_init, edge_index, test_idx)
            x = self.fagcn(x, x_init, edge_index, test_idx)
            x = self.fagcn(x, x_init, edge_index, test_idx)
        x = F.relu(x)
        x_out = self.pred(x)
        
        calib = - torch.max(x_out, 1)[0] + torch.topk(x_out, 2)[0][:, 1]
            
        return F.log_softmax(x_out, dim=1), torch.mean(calib), F.softmax(x_out, dim=1)


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
        
        out, calib, _ = model(data, data.edge_index, 0, 0)
    
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + calib * lam * use_calib
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, _, xs = model(data, data.edge_index, 0, 1)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                        
            if valid > best_valid:
                best_valid = valid
                best_value = acc
                print('%.3f' % best_value)
                save = xs
    
