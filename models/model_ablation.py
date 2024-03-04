import torch.nn
from torch import nn
from torch.nn import Linear, Sigmoid, Tanh, Softmax, GELU, ReLU
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
from torch_geometric.nn import GCNConv, TransformerConv, GATConv
from torch_geometric.nn import GAE


def getactfn(actfn = 'relu'):
    if actfn == 'relu':
        return ReLU()
    elif actfn == 'gelu':
        return GELU()
    elif actfn == 'tanh':
        return Tanh()
    
def getnorm(norm = 'layernorm', norm_shape =0):
    if norm == 'layernorm':
        return LayerNorm(norm_shape)
    elif norm == 'batchnorm':
        return BatchNorm1d(norm_shape)
    
class MLP(nn.Module):
    def __init__(self, **kwargs):

        layers_des = kwargs.get('layers', None)
        self.aggregate = kwargs.get('aggregate', None)
        in_dropout = kwargs.get('in_dropout', 0.0)

        output_dim = layers_des[-1][1]
        input_dim = layers_des[0][0]
        

        super(MLP, self).__init__()

        self.in_dropout = nn.Dropout(in_dropout)

        self.layers = nn.ModuleList()
        self.layer_inputs = []

        for pos, layer_dims in enumerate(layers_des):
            tmp = OrderedDict()
            tmp['mlp_{}'.format(pos)] = nn.Linear(layer_dims[0], layer_dims[1], bias=layer_dims[2])
            if layer_dims[4] != 'none':
                tmp['norm_{}'.format(pos)] =  getnorm(layer_dims[4], layer_dims[1])
            if layer_dims[3] != 'none':
                tmp['act_fn_{}'.format(pos)] = getactfn(layer_dims[3])
            if layer_dims[5] != 'none':
                tmp['dropout_{}'.format(pos)] = nn.Dropout(p=layer_dims[5])
            self.layer_inputs.append(layer_dims[6])
            self.layers.append(nn.Sequential(tmp))


        self.skip_1 = torch.nn.Linear(input_dim, output_dim)
        self.skip_final = torch.nn.Linear(output_dim*2, output_dim)
        
        
    def forward(self, features, batches=None, residual=True):

        features_array = []

        features = self.in_dropout(features)

        if self.aggregate == 'start':
            features = net_utils.get_pool(pool_type='mean')(features, batches)

        features_array.append(features)

        for lay, layer_input_sizes in zip(self.layers, self.layer_inputs):
            x = []
            for i in layer_input_sizes:
                x.append(features_array[i])
            x = torch.cat(x, 1)
            x = lay(x)
            features_array.append(x)

        if residual == True:
            x_skip = self.skip_1(features_array[0])
            x = torch.cat([x, x_skip], 1)
            x = self.skip_final(x)

        if self.aggregate == 'end':
            x = net_utils.get_pool(pool_type='mean')(x, batches)

        return x


class TFun_submodel(nn.Module):
    def __init__(self, **kwargs):
        super(TFun_submodel, self).__init__()

        self.ont = kwargs['ont']
        self.device = kwargs['device']
        self.sub_model = kwargs['sub_model']
        self.full = kwargs.get('full', False)
        self.group = kwargs['group']
        self.sigmoid = Sigmoid()

        if self.sub_model == 'interpro':
            interpro_layers = getattr(config, "interpro_layers_{}_{}".format(self.group, self.ont))
            self.interpro_mlp = MLP(layers = interpro_layers, in_dropout=0.1)
            
        if self.sub_model == 'msa_1b':
            msa_layers = getattr(config, "msa_layers_{}_{}".format(self.group, self.ont))
            self.msa_mlp = MLP(layers = msa_layers, in_dropout=0.0)
            
        if self.sub_model == 'esm2_t48':
            esm_layers = getattr(config, "esm_layers_{}_{}".format(self.group, self.ont))
            self.esm_mlp = MLP(layers = esm_layers, in_dropout=0.1)
           

    def forward(self, data):
        if self.sub_model == 'interpro':
            interpro_out = self.interpro_mlp(data.to(self.device), residual=True)
            if self.full:
                return interpro_out
            else:
                return self.sigmoid(interpro_out)
        
        elif self.sub_model == 'esm2_t48':
            esm_out = self.esm_mlp(data.to(self.device), residual=True)
            if self.full:
                return esm_out
            else:
                return self.sigmoid(esm_out)
        
        elif self.sub_model == 'msa_1b':
            msa_out = self.msa_mlp(data.to(self.device), residual=True)
            if self.full:
                return msa_out
            else:
                return self.sigmoid(msa_out)


class LabelEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features, gnn="GCN"):
        super(LabelEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        if gnn == "GCN":
            self.conv1 = GCNConv(self.in_channels, 2 * self.out_channels, cached=True)
            self.conv2 = GCNConv(2 * self.out_channels, self.out_channels, cached=True) 
        elif gnn == "GAT": 
            self.conv1 = GATConv(self.in_channels, 2*self.out_channels, heads=1, concat=True)
            self.conv2 = GATConv(2 * self.out_channels, self.out_channels, heads=1, concat=True) 
        elif gnn == "TRANSFORMER":
            self.conv1 = TransformerConv(in_channels, 2 * self.out_channels, 
                                         heads=1, beta=True, concat=True)
            self.conv2 = TransformerConv(2 * self.out_channels, self.out_channels, 
                                         heads=1, beta=True, concat=True)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn = BatchNorm1d(2 * self.out_channels)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        if self.features == 'biobert' or self.features == 'bert':
            x = self.bn(x)
            x = self.tanh(x)
            # x = self.relu(x)
        elif self.features == 'x':
            x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


def load_submodel(ckp_pth, **kwargs):
        x = TFun_submodel(**kwargs)
        if kwargs['load_weights']:
            x = load_ckp(ckp_pth.format(kwargs['sub_model'].format(kwargs['group'])), 
                         x, optimizer=None, lr_scheduler=None, best_model=False, model_only=True)
        return x


def map_device_location(device):
    if device == "cpu":
        return 
    elif device == "cuda:0":
        return 
    elif device == "cuda:1":
        return

class Attention(nn.Module):

    def __init__(self, **kwargs):
        super(Attention, self).__init__()

        label_dimension = kwargs['label_dimension']
        embedding_size = kwargs['embedding_size']
        concat_hidden = kwargs['concat_hidden']
        self.heads = kwargs.get('heads', 1)
        out_proj = kwargs['out_proj']

         # proteins as queries
        self.q = nn.Linear(concat_hidden, embedding_size)
        # GO terms as keys & values
        self.k = nn.Linear(label_dimension, embedding_size)
        self.v = nn.Linear(label_dimension, embedding_size)
        self.final = nn.Linear(embedding_size, out_proj)
        self.softmax = Softmax(dim=-1)


    def forward(self, x_1, x_2):
        q = self.q(x_1)
        k = self.k(x_2)
        v = self.v(x_2)

        dk = k.size()[-1]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(dk)
        attention = self.softmax(scores)
        values = torch.matmul(attention, v)

        sum_out = values + q
        proj_out = self.final(sum_out)
        return proj_out, attention


def scatter_accross(ont, shape, concat, **kwargs):

    device = kwargs['esm_freq_out'].device
    batch_size = kwargs['esm_freq_out'].shape[0]
    freq_indicies = kwargs['freq_indicies'].repeat(batch_size, 1)
    rare_indicies = kwargs['rare_indicies'].repeat(batch_size, 1)
    

    esm_out = torch.zeros(batch_size, shape, dtype=kwargs['esm_freq_out'].dtype, 
                          device=device)
    esm_out = esm_out.scatter_(1, freq_indicies, kwargs['esm_freq_out'])
    esm_out = esm_out.scatter_(1, rare_indicies, kwargs['esm_rare_out'])

    msa_out = torch.zeros(batch_size, shape, dtype=kwargs['msa_freq_out'].dtype, 
                          device=device)
    msa_out = msa_out.scatter_(1, freq_indicies, kwargs['msa_freq_out'])
    msa_out = msa_out.scatter_(1, rare_indicies, kwargs['msa_rare_out'])

    interpro_out = torch.zeros(batch_size, shape, dtype=kwargs['interpro_freq_out'].dtype, 
                          device=device)
    interpro_out = interpro_out.scatter_(1, freq_indicies, kwargs['interpro_freq_out'])
    interpro_out = interpro_out.scatter_(1, rare_indicies, kwargs['interpro_rare_out'])


    # add last bacth of biological process
    if ont == 'bp':
        rare_indicies_2 = kwargs['rare_indicies_2'].repeat(batch_size, 1)

        esm_out = esm_out.scatter_(1, rare_indicies_2, kwargs['esm_rare_out_2'])
        msa_out = msa_out.scatter_(1, rare_indicies_2, kwargs['msa_rare_out_2'])
        interpro_out = interpro_out.scatter_(1, rare_indicies_2, kwargs['interpro_rare_out_2'])

    if concat == 'concat': # just concatenate
        x = torch.cat((esm_out, msa_out, interpro_out), 1)
    elif concat == "mean":
        _batches = torch.arange(batch_size)
        batches = torch.cat((_batches, _batches, _batches), 0).to(device)
        x = torch.cat((esm_out, msa_out, interpro_out), 0)
        x = net_utils.get_pool(pool_type='mean')(x, batches)
    elif concat == "max":
        _batches = torch.arange(batch_size)
        batches = torch.cat((_batches, _batches, _batches), 0).to(device)
        x = torch.cat((esm_out, msa_out, interpro_out), 0)
        x = net_utils.get_pool(pool_type='max')(x, batches)

    return x


class TFun(nn.Module):
    def __init__(self, **kwargs):
        super(TFun, self).__init__()

        self.ont = kwargs['ont']
        self.device = kwargs['device']
        self.full_indicies = kwargs['full_indicies'].to(self.device)
        self.freq_indicies = kwargs['freq_indicies'].to(self.device)
        self.rare_indicies = kwargs['rare_indicies'].to(self.device)
        self.rare_indicies_2 =  kwargs['rare_indicies_2'].to(self.device) if self.ont =="bp" else None
        self.out_shape = getattr(config, "{}_out_shape".format(self.ont))
        self.label_features = kwargs['label_features']
        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        ckp_pth = CONSTANTS.ROOT_DIR + "{}/models/".format(self.ont) + "{}/" 
        kwargs['full'] = True
        kwargs['load_weights'] = False


        # esm submodels
        kwargs['sub_model'] = 'esm2_t48'
        kwargs['group'] = 'freq'
        self.esm_freq_mlp = load_submodel(ckp_pth, **kwargs)
        
        kwargs['sub_model'] = 'esm2_t48'
        kwargs['group'] = 'rare'
        self.esm_rare_mlp = load_submodel(ckp_pth, **kwargs)


        # interpro
        kwargs['sub_model'] = 'interpro'
        kwargs['group'] = 'freq'
        self.interpro_freq_mlp = load_submodel(ckp_pth, **kwargs)

        kwargs['sub_model'] = 'interpro'
        kwargs['group'] = 'rare'
        self.interpro_rare_mlp = load_submodel(ckp_pth, **kwargs)
        
        
        # msa
        kwargs['sub_model'] = 'msa_1b'
        kwargs['group'] = 'freq'
        self.msa_freq_mlp = load_submodel(ckp_pth, **kwargs)

        kwargs['sub_model'] = 'msa_1b'
        kwargs['group'] = 'rare'
        self.msa_rare_mlp = load_submodel(ckp_pth, **kwargs)


        # Add extra for BP
        if self.ont == 'bp':
            kwargs['group'] = 'rare_2'

            kwargs['sub_model'] = 'esm2_t48'
            self.esm_rare_mlp_2 = load_submodel(ckp_pth, **kwargs)

            kwargs['sub_model'] = 'interpro'
            self.interpro_rare_mlp_2 = load_submodel(ckp_pth, **kwargs)

            kwargs['sub_model'] = 'msa_1b'
            self.msa_rare_mlp_2 = load_submodel(ckp_pth, **kwargs)



        if self.label_features == 'biobert':
            self.joint_embedding = Attention(concat_hidden=self.out_shape*3, 
                                          label_dimension=768,
                                           embedding_size=512,
                                           out_proj = self.out_shape)
        elif self.label_features == 'x':
            self.joint_embedding = Attention(concat_hidden=self.out_shape*3, 
                                           label_dimension=self.out_shape, # no of classes
                                           embedding_size=768,
                                           out_proj = self.out_shape)
        elif self.label_features == 'gcn':
            self.label_embedding = GAE(LabelEncoder(768, 1024, features='biobert'))
            self.label_embedding = load_ckp(ckp_pth.format("label"), self.label_embedding, optimizer=None, lr_scheduler=None, best_model=True, model_only=True)
            self.joint_embedding = Attention(concat_hidden=self.out_shape*3, 
                                           label_dimension=1024, 
                                           embedding_size=256 if self.ont == 'bp' else 256, #768,#2048,
                                           out_proj = self.out_shape)
        elif self.label_features == 'linear':
            self.joint_embedding_1 = Attention(concat_hidden=self.out_shape*3, 
                                            label_dimension=768,
                                            embedding_size=384,
                                            out_proj = self.out_shape)
            
            self.joint_embedding_2 = Attention(concat_hidden=self.out_shape*3, 
                                            label_dimension=self.out_shape, # no of classes
                                            embedding_size=384,
                                            out_proj = self.out_shape)
            self.final = nn.Linear(self.out_shape*2, self.out_shape)

        self.graph_path = CONSTANTS.ROOT_DIR + '{}/graph.pt'.format(self.ont)
        # self.label_embedd_data = torch.load(self.graph_path, map_location=map_device_location()lambda storage, loc: storage.cuda(1))
        self.label_embedd_data = torch.load(self.graph_path, map_location=torch.device(self.device))
    
        self.gelu = nn.GELU()
        self.sigmoid = Sigmoid()
    

    def forward(self, data):
        esm_freq_out = self.esm_freq_mlp(data[0].to(self.device))
        esm_rare_out = self.esm_rare_mlp(data[0].to(self.device))

        msa_freq_out = self.msa_freq_mlp(data[1].to(self.device))
        msa_rare_out = self.msa_rare_mlp(data[1].to(self.device))

        interpro_freq_out = self.interpro_freq_mlp(data[3].to(self.device))
        interpro_rare_out = self.interpro_rare_mlp(data[3].to(self.device))

        extraargs = {
            'esm_freq_out': esm_freq_out, 'esm_rare_out': esm_rare_out,
            'msa_freq_out': msa_freq_out, 'msa_rare_out': msa_rare_out,
            'interpro_freq_out': interpro_freq_out, 'interpro_rare_out': interpro_rare_out, 
            'freq_indicies': self.freq_indicies, 'rare_indicies': self.rare_indicies

        }

        if self.ont == 'bp':
            esm_rare_out_2 = self.esm_rare_mlp_2(data[0].to(self.device))
            msa_rare_out_2 = self.msa_rare_mlp_2(data[1].to(self.device))
            interpro_rare_out_2 = self.interpro_rare_mlp_2(data[3].to(self.device))

            extraargs['esm_rare_out_2'] = esm_rare_out_2
            extraargs['msa_rare_out_2'] = msa_rare_out_2
            extraargs['interpro_rare_out_2'] = interpro_rare_out_2
            extraargs['rare_indicies_2'] = self.rare_indicies_2

    
        if self.label_features == 'mean':
            x = scatter_accross(self.ont, self.out_shape, concat="mean", **extraargs)
        elif self.label_features == 'max':
            x = scatter_accross(self.ont, self.out_shape, concat="max", **extraargs)
        else:
            x = scatter_accross(self.ont, self.out_shape, concat="concat", **extraargs)

            if self.label_features == 'biobert':
                z = self.label_embedd_data['biobert'].to(self.device)
                x, att = self.joint_embedding(x, z)
            elif self.label_features == 'x':
                z = self.label_embedd_data['x'].to(self.device)
                x, att = self.joint_embedding(x, z)
            elif self.label_features == 'gcn':
                z = self.label_embedding.encode(self.label_embedd_data['biobert'].to(self.device), self.label_embedd_data.edge_index.to(self.device)).to(self.device)
                x, att = self.joint_embedding(x, z)
            elif self.label_features == 'linear':
                z_1 = self.label_embedd_data['biobert'].to(self.device)
                z_2 = self.label_embedd_data['x'].to(self.device)
               
                x_1 = self.joint_embedding_1(x, z_1)
                x_2 = self.joint_embedding_2(x, z_2)
                x = torch.cat((x_1, x_2), 1)
                x, att = self.final(x)

        x = self.sigmoid(x)
        return x, att



    

