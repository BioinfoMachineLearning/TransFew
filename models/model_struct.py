import torch.nn
from torch import nn
from torch.nn import Linear, Sigmoid
from torch.nn.modules.module import Module
from torchsummary import summary
from Utils import load_ckp, pickle_load
import torch.nn.functional as F
from models import net_utils
import models.config as config
import math
import CONSTANTS
from torch.nn import LayerNorm, BatchNorm1d
from collections import OrderedDict
from torch_geometric.nn import GCNConv, TransformerConv, GATv2Conv
from torch_geometric.nn import GAE


def getactfn(actfn = 'relu'):
    if actfn == 'relu':
        return nn.ReLU()
    elif actfn == 'gelu':
        return nn.GELU()
    elif actfn == 'tanh':
        return nn.Tanh()
    
def getnorm(norm = 'layernorm', norm_shape =0):
    if norm == 'layernorm':
        return LayerNorm(norm_shape)
    elif norm == 'batchnorm':
        return BatchNorm1d(norm_shape)


class TFun(nn.Module):
    def __init__(self, **kwargs):
        super(TFun, self).__init__()

        self.ont = kwargs['ont']
        self.device = kwargs['device']
        self.indicies = kwargs['indicies']
        self.load_weights = True #kwargs['load_weights']
        self.out_shape = getattr(config, "{}_out_shape".format(self.ont))
        self.label_features = kwargs['label_features']
        self.dropout = nn.Dropout(0.1)

        self.layer1 = nn.Linear(1024, 1200)
        self.layer2 = nn.Linear(1200, 1500)
        self.layer3 = nn.Linear(1500, 2000)
        self.final = nn.Linear(2000, self.out_shape)

    
        self.gelu = nn.GELU()
        self.sigmoid = Sigmoid()
           

    def forward(self, x):
        x = self.layer1(x.to(self.device))
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.gelu(x)
        x = self.layer3(x)
        x = self.gelu(x)
        x = self.final(x)
        x = torch.index_select(x, 1, self.indicies)
        x = self.sigmoid(x)
        return x
    

class TFunSequence(nn.Module):
    def __init__(self, **kwargs):
        super(TFunSequence, self).__init__()

        self.ont = kwargs['ont']
        self.device = kwargs['device']
        self.indicies = kwargs['indicies']
        self.load_weights = True #kwargs['load_weights']
        self.out_shape = getattr(config, "{}_out_shape".format(self.ont))
        self.label_features = kwargs['label_features']
        self.dropout = nn.Dropout(0.1)

        
        # Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=4),
            num_layers=6
        )

        # Token embeddings
        # self.embedding = nn.Embedding(100, 512)

        # Positional encoding
        self.positional_encoding = nn.Embedding(1024, 512)  # Assuming sequences of max length 1000

        # Fully connected layer for classification
        self.fc = nn.Linear(512, self.out_shape)

        self.sigmoid = Sigmoid()
           

    def forward(self, x):


        # Add positional encoding to the input
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(x.size(0), -1).to(self.device)
        positions = self.positional_encoding(positions)
        #print(self.positional_encoding(positions).shape)
        x = x + positions

        print(positions.shape)


        exit()

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), (1,)).view(x.size(0), -1)

        # Classification layer
        x = self.fc(x)
        x = self.sigmoid(x)

        return x