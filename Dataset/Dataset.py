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
            self.data = kwargs.get('proteins', [])
       
        super().__init__(transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return CONSTANTS.ROOT_DIR + "data/raw"

    @property
    def processed_dir(self) -> str:
        return CONSTANTS.ROOT_DIR + "data/processed"

    @property
    def raw_file_names(self):
        if self.split != 'selected':
            x = list(pickle_load(CONSTANTS.ROOT_DIR + "{}/all_proteins".format(self.ont)))
            self.data = x
        return self.data

    @property
    def processed_file_names(self):
        data = os.listdir(self.processed_dir)
        data = [i.split(".")[0] for i in data]
        return data

    def download(self):
        pass

    def process(self):
        raw = set(self.raw_file_names)
        processed = set(self.processed_file_names)
        remain = raw - processed

        print("Raw Data {} --- Processed Data {} --- Remaining {}".\
                  format(len(raw), len(processed), len(remain)))
        
        if len(remain) >  0:

            print("Loading interpro")
            # Interpro Data
            interpro = Interpro(ont='mf')
            # mf_interpro_data, mf_interpro_sig, _ = interpro.get_interpro_ohe_data()
            mf_interpro_data, mf_interpro_sig, _ = interpro.get_interpro_test()
            print("loaded mf")

            interpro = Interpro(ont='cc')
            # cc_interpro_data, cc_interpro_sig, _ = interpro.get_interpro_ohe_data()
            cc_interpro_data, cc_interpro_sig, _ = interpro.get_interpro_test()
            print("loaded cc")

            interpro = Interpro(ont='bp')
            # bp_interpro_data, bp_interpro_sig, _ = interpro.get_interpro_ohe_data()
            bp_interpro_data, bp_interpro_sig, _ = interpro.get_interpro_test()
            print("loaded bp")

            for num, protein in enumerate(remain):
                print(protein, num, len(remain))

                esm_msa1b = torch.load("/bmlfast/frimpong/shared_function_data/esm_msa1b/{}.pt".format(protein))['representations_12'].squeeze(0).cpu().detach()#.to('cpu', dtype=torch.float32) #.detach().numpy()
                esm2_t48 = torch.load("/bmlfast/frimpong/shared_function_data/esm2_t48/{}.pt".format(protein))['mean_representations'][48].view(1, -1).cpu()

                mf_interpro_prot = torch.Tensor(np.array(mf_interpro_data.get(protein, [0] * len(mf_interpro_sig)))).view(1, -1)
                cc_interpro_prot = torch.Tensor(np.array(cc_interpro_data.get(protein, [0] * len(cc_interpro_sig)))).view(1, -1)
                bp_interpro_prot = torch.Tensor(np.array(bp_interpro_data.get(protein, [0] * len(bp_interpro_sig)))).view(1, -1)


                data = HeteroData()

                data['interpro_mf'].x = mf_interpro_prot
                data['interpro_cc'].x = cc_interpro_prot
                data['interpro_bp'].x = bp_interpro_prot

                data['esm2_t48'].x = esm2_t48
                data['esm_msa1b'].x = esm_msa1b

                data.protein = protein

                torch.save(data, osp.join(self.processed_dir, f'{protein}.pt'))


    def len(self):
        return len(self.raw_file_names)


    def get(self, idx):
        rep = self.raw_file_names[idx]
        return rep