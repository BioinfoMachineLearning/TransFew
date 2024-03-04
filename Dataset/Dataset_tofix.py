import math
import os
import pickle
import subprocess
import numpy as np
import pandas as pd
import torch
import os.path as osp
import CONSTANTS
from torch_geometric.data import Dataset
from Classes.Diamond import Diamond
from torch_geometric.data import Data, HeteroData
from Classes.Interpro import Interpro
from Classes.STRING import STRING
from Utils import pickle_load, readlines_cluster
import random


class TransFunDataset(Dataset):
    """
        Creates a dataset from a list of PDB files.
        :param file_list: path to LMDB file containing dataset
        :type file_list: list[Union[str, Path]]
        :param transform: transformation function for data augmentation, defaults to None
        :type transform: function, optional
        """

    def __init__(self, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.ont = kwargs.get('ont', None)
        self.split = kwargs.get('split', None)

        if self.split == 'selected':
            self.data = kwargs.get('proteins', 'proteins')
        else:
            self.cluster = readlines_cluster(CONSTANTS.ROOT_DIR + "{}/mmseq_0.6/final_clusters.csv".format(self.ont))
            self.indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/{}_indicies".format(self.ont, self.split))


        super().__init__(transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return CONSTANTS.ROOT_DIR + "/data/raw"

    @property
    def processed_dir(self) -> str:
        return CONSTANTS.ROOT_DIR + "/data/processed"

    @property
    def raw_file_names(self):
        
        if self.split == 'selected':
            return self.data
        else:
            x = list(pickle_load(CONSTANTS.ROOT_DIR + "train_validation")[self.ont][self.split])
            self.data = x
            return x


    @property
    def processed_file_names(self):
        data = os.listdir(self.processed_dir)
        return data

    def download(self):
        pass


    def process(self):

            raw = set(self.raw_file_names)
            processed = set(self.processed_file_names)
            remain = raw - processed
        
            if len(remain) > 0:
                print("Raw Data {} --- Processed Data {} --- Remaining {}".\
                  format(len(raw), len(processed), len(remain)))

                # String Data
                ppi = STRING()
                ppi_data_cc = ppi.get_string_neighbours(ontology='cc', confidence=0.4, recreate=False)
                print("Number of nodes in ppi", len(ppi_data_cc))

                ppi_data_mf = ppi.get_string_neighbours(ontology='mf', confidence=0.4, recreate=False)
                print("Number of nodes in ppi", len(ppi_data_mf))

                ppi_data_bp = ppi.get_string_neighbours(ontology='bp', confidence=0.4, recreate=False)
                print("Number of nodes in ppi", len(ppi_data_bp))

                labels_cc = pickle_load(CONSTANTS.ROOT_DIR + "cc/labels")

                labels_mf = pickle_load(CONSTANTS.ROOT_DIR + "mf/labels")

                labels_bp = pickle_load(CONSTANTS.ROOT_DIR + "bp/labels")

                for num, protein in enumerate(remain):
                    print(protein, num, len(remain))

                    xx = torch.load("/home/fbqc9/Workspace/DATA/data/processed1/{}.pt".format(protein))

                    # STRING
                    string_neighbours = ppi_data_cc.get(protein, [])
                    string_cc = []
                    for neighbour in string_neighbours:
                        if neighbour == protein:
                            pass
                        else:
                            if neighbour in labels_cc:
                                string_cc.append(np.array(labels_cc[neighbour], dtype=int))

                    string_neighbours = ppi_data_mf.get(protein, [])
                    string_mf = []
                    for neighbour in string_neighbours:
                        if neighbour == protein:
                            pass
                        else:
                            if neighbour in labels_mf:
                                string_mf.append(np.array(labels_mf[neighbour], dtype=int))

                    string_neighbours = ppi_data_bp.get(protein, [])
                    string_bp = []
                    for neighbour in string_neighbours:
                        if neighbour == protein:
                            pass
                        else:
                            if neighbour in labels_bp:
                                string_bp.append(np.array(labels_bp[neighbour], dtype=int))

                    if len(string_cc) > 0:
                        string_cc = torch.Tensor(np.vstack(string_cc))
                    else:
                        string_cc = torch.Tensor(np.array([0] * 2957)).unsqueeze(0)

                    if len(string_mf) > 0:
                        string_mf = torch.Tensor(np.vstack(string_mf))
                    else:
                        string_mf = torch.Tensor(np.array([0] * 7224)).unsqueeze(0)

                    if len(string_bp) > 0:
                        string_bp = torch.Tensor(np.vstack(string_bp))
                    else:
                        string_bp = torch.Tensor(np.array([0] * 21285)).unsqueeze(0)



                    xx['string_cc'].x = string_cc
                    xx['string_mf'].x = string_mf
                    xx['string_bp'].x = string_bp

                    del xx['interpro']


                    assert len(xx['esm2_t48'].x.shape) == len(xx['esm_msa1b'].x.shape) \
                          == len(xx['diamond_cc'].x.shape) == len(xx['diamond_mf'].x.shape) \
                            == len(xx['diamond_bp'].x.shape) == len(xx['string_cc'].x.shape) \
                                == len(xx['string_mf'].x.shape) == len(xx['string_bp'].x.shape) \
                                    == len(xx['interpro_cc'].x.shape) == len(xx['interpro_mf'].x.shape)\
                                         == len(xx['interpro_bp'].x.shape) == 2

                    torch.save(xx, osp.join(self.processed_dir, f'{protein}.pt'))

   


    def len(self):
        if self.split == "train":
            return len(self.indicies)
        else: 
            return len(self.raw_file_names)


    def get(self, idx):
        if self.split == "train":
            cluster_index = self.indicies[idx]
            rep = random.sample(self.cluster[cluster_index], 1)[0]
            assert rep in set(self.raw_file_names)
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))
        else: 
            rep = self.data[idx]
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))