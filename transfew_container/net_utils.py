import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class FC(nn.Module):
    def __init__(self, in_features, out_features, act_fun='relu', bnorm=True):
        super(FC, self).__init__()
        bias = False if bnorm else True
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.act_fun = get_act_fun(act_fun)
        self.bn = nn.BatchNorm1d(out_features, momentum=0.1) if bnorm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_fun is not None:
            x = self.relu(x)
        return x


class BNormActFun(nn.Module):
    def __init__(self, in_features, act_fun='relu', momentum=0.1, bnorm=True):
        super(BNormActFun, self).__init__()

        self.act_fun = get_act_fun(act_fun)
        self.bn = nn.BatchNorm1d(in_features, momentum=momentum) if bnorm else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        if self.act_fun is not None:
            x = self.act_fun(x)
        return x


class MLP(nn.Module):
    # [(in_features, out_features, bnorm, act_fun, dropuout)]
    def __init__(self, layers_data: list):
        super().__init__()
        torch.manual_seed(12345)
        self.layers = nn.ModuleList()
        for layer in layers_data:
            # input_size output_size bias
            self.layers.append(nn.Linear(layer[0], layer[1], not layer[2]))
            if layer[2]:
                self.layers.append(nn.BatchNorm1d(layer[1], momentum=0.1))
            # activation function
            if layer[3] is not None:
                self.layers.append(get_act_fun(layer[3]))
            # dropout
            if layer[4] is not None:
                self.layers.append(nn.Dropout(p=layer[4]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_pool(pool_type='max'):
    if pool_type == 'mean':
        return global_mean_pool
    elif pool_type == 'add':
        return global_add_pool
    elif pool_type == 'max':
        return global_max_pool


def get_act_fun(act_fun):
    if act_fun == 'relu':
        return nn.ReLU()
    elif act_fun == 'tanh':
        return nn.Tanh()
    elif act_fun == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        return None
