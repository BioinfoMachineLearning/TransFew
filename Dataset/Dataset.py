import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from Utils import pickle_load
import CONSTANTS

class TransFewDataset(Dataset):
    def __init__(self, data_pth=None,  submodel=None):

        self.submodel = submodel

        data = pickle_load(data_pth)

        labels = data['labels']
        labels = torch.cat(labels, dim=0)

        self.labels = labels
        self.esm_features = data['esm2_t48']
        self.msa_features = data['msa_1b']
        #self.diamond_features = data['diamond']
        self.interpro_features = data['interpro']
        #self.string_features = data['string']
        
        
    def __getitem__(self, index):

        esm = self.esm_features[index]
        msa = self.msa_features[index]
        # diamond = self.diamond_features[index]
        interpro = self.interpro_features[index]
        # string = self.string_features[index]
        label = self.labels[index]

        if self.submodel == 'esm2_t48':
            return esm, label
        elif self.submodel == 'msa_1b':
            return msa, label
        elif self.submodel == 'interpro':
            return interpro, label
        elif self.submodel == 'full':
            # return esm, msa, diamond, interpro, string, label
            return esm, msa, interpro, label
        '''elif self.submodel == 'diamond':
            return diamond, label
        elif self.submodel == 'string':
            return string, label'''
    
    def __len__(self):
        return len(self.labels)


class TestDataset(Dataset):
    def __init__(self, data_pth=None, submodel=None):

        self.submodel = submodel

        data = pickle_load(data_pth)

        self.proteins = data['protein']
        self.esm_features = data['esm2_t48']
        self.msa_features = data['msa_1b']
        self.diamond_features = data['diamond']
        self.interpro_features = data['interpro']
        
    def __getitem__(self, index):

        esm = self.esm_features[index]
        msa = self.msa_features[index]
        diamond = self.diamond_features[index]
        interpro = self.interpro_features[index]
        proteins = self.proteins[index]

        

        if self.submodel == 'esm2_t48':
            return esm, proteins
        elif self.submodel == 'msa_1b':
            return msa, proteins
        elif self.submodel == 'diamond':
            return diamond, proteins
        elif self.submodel == 'interpro':
            return interpro, proteins
        elif self.submodel == 'full':
            return esm, msa, diamond, interpro, proteins
    
    def __len__(self):
        return len(self.proteins)
    


class PredictDataset(Dataset):
    def __init__(self, data=None):

        self.esm_features = data['esm2_t48']
        self.proteins = data['protein']
        
        
    def __getitem__(self, index):

        esm = self.esm_features[index]
        proteins = self.proteins[index]

        return esm, proteins
    
    def __len__(self):
        return len(self.proteins)