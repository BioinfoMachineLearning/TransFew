import os
import numpy as np
import pandas as pd
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import CONSTANTS
from Classes.Diamond import Diamond
from Classes.Fasta import Fasta
from Classes.Interpro import Interpro, create_indicies
from Classes.STRING import STRING
from Utils import count_proteins, get_proteins_from_fasta, pickle_load, pickle_save
import urllib.request



onts = ['cc', 'bp', 'mf']
sess = ['validation', 'train']


key_val = pickle_load("cc_string_comps").key_val
data = torch.load("cc_string_comp.pt")
data = data.cpu()

# res = dict((v,k) for k,v in _data.items())

# you = {}
# for i, j in enumerate(data):
#     you[res[i]] = j
# print(len(you))


for ont in onts[0:1]:
    print(ont)

    for s in sess:
        print(s)

        dt = list(pickle_load(CONSTANTS.ROOT_DIR + "{}/{}_proteins".format(ont, s)))

        indicies = torch.tensor([key_val[i] for i in dt])


        bn =  torch.index_select(data, 0, indicies)
        bn = bn.tolist()

        labels = pickle_load(CONSTANTS.ROOT_DIR + "{}/labels".format(ont))

        store = {'labels': [],
                  'string': bn
                  }

        for i in dt:
            label = torch.tensor(labels[i], dtype=torch.float32).view(1, -1)

            store['labels'].append(label)


        pickle_save(store, CONSTANTS.ROOT_DIR + "{}/{}_data_2".format(ont, s))




exit()

onts = ['cc', 'bp', 'mf']
sess = ['train', 'validation']


for ont in onts:
    print(ont)

    for s in sess:
        
        dt = list(pickle_load(CONSTANTS.ROOT_DIR + "{}/{}_proteins".format(ont, s)))

        labels = pickle_load(CONSTANTS.ROOT_DIR + "{}/labels".format(ont))

        store = {'labels': [],
                  'esm2_t48': [],
                  'msa_1b': [],
                  'interpro': [],
                  'diamond': [],
                  'string': [],
                  'protein': []
                  }
        
        for i in dt:
            print("{}, {}, {}".format(ont, s, i))
            tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))
            esm = tmp['esm2_t48'].x.squeeze(0)
            msa = torch.mean(tmp['esm_msa1b'].x, dim=0).detach()#.unsqueeze(0)
            diamond = tmp['diamond_{}'.format(ont)].x
            diamond = torch.mean(diamond, dim=0)#.unsqueeze(0)
            interpro = tmp['interpro_{}'.format(ont)].x.squeeze(0)
            string_data = tmp['string_{}'.format(ont)].x
            string_data = torch.mean(string_data, dim=0)#.unsqueeze(0)
            label = torch.tensor(labels[i], dtype=torch.float32).view(1, -1)


            store['labels'].append(label)
            store['esm2_t48'].append(esm)
            store['msa_1b'].append(msa)
            store['diamond'].append(diamond)
            store['interpro'].append(interpro)
            store['string'].append(string_data)
            store['protein'].append(i)


        pickle_save(store, CONSTANTS.ROOT_DIR + "{}/{}_data".format(ont, s))



exit()
# device = 'cuda:1'
train_data = pickle_load("com_data/{}.data_{}".format('cc', 'train'))
print(train_data['esm2_t48'].shape)
print(train_data['msa_1b'].shape)
print(train_data['diamond'].shape)
print(train_data['interpro'].shape)
print(train_data['string'].shape)
print(train_data['labels'].shape)


# # labels = torch.cat(labels, dim=0).to(device)
# # labels = torch.index_select(labels, 1, term_indicies)

# msa_features = train_data['msa_1b']

# for i in msa_features:
#     print(i)
#     exit()

# print(type(msa_features))
# exit()
# msa_features = torch.cat(msa_features, dim=0).to(device)



# esm_features = train_data['esm2_t48']
# esm_features = torch.cat(esm_features, dim=0).to(device)




# data = pickle_load("com_data/{}.data_{}".format(self.ont, self.session))

# labels = data['labels']
# labels = torch.cat(labels, dim=0).to(device)
# labels = torch.index_select(labels, 1, term_indicies)

# esm_features = train_data[args.submodel]
# esm_features = torch.cat(esm_features, dim=0).to(device)


# exit()

# print(esm_features.shape, msa_features.shape)
# print(esm_features.dtype, msa_features.dtype)
# print(esm_features.device, msa_features.device)
# print(esm_features[0].dtype, msa_features[0].dtype)
