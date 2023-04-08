import os.path as osp

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Dataset, download_url
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from Utils import is_file
import CONSTANTS
from Classes.Diamond import Diamond
from preprocessing.utils import pickle_load


class DiamondDataset(InMemoryDataset):

    def __init__(self, split: str = "random", ont: str = "cc"):

        self.ont = ont
        self.name = "diamond_{}".format(ont)
        super().__init__()
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split

        if split == 'preset':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])
        elif split == 'random':
            pass

    @property
    def processed_file_names(self) -> str:
        return '{}_data.pt'.format(self.name)

    @property
    def processed_dir(self) -> str:
        return CONSTANTS.ROOT_DIR + 'datasets/diamond'

    def process(self):
        kwargs = {
            'fasta_file': CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta"
        }
        diamond = Diamond(CONSTANTS.ROOT_DIR + "diamond", **kwargs)
        G = diamond.get_graph()
        nodes = set(G.nodes)



        tmp = pickle_load(CONSTANTS.ROOT_DIR + "datasets/training_validation")
        train_val = tmp[self.ont]['train'].union(tmp[self.ont]['valid'])

        remove = nodes.difference(train_val)
        print("Total nodes {}; Removing {}; Retaining {}".format(len(nodes), len(remove), len(train_val)))
        G.remove_nodes_from(remove)
        nodes = list(G.nodes)

        embeddings = pickle_load(CONSTANTS.ROOT_DIR + "embedding/esm_36_all")
        embeddings = {key: embeddings[key] for key in nodes if key in embeddings}


        node_features = {}
        layer = 36
        test_nodes = set()
        for pos, node in enumerate(nodes):
            if node in embeddings:
                node_features[node] = {'{}'.format(layer): embeddings[node]}
                test_nodes.add(node)
            else:
                G.remove_node(node)

        print("Diamond Graph with {} nodes".format(len(G.nodes)))
        nodes = list(G.nodes)

        y = pickle_load(CONSTANTS.ROOT_DIR + "datasets/labels")[self.ont]
        y = np.array([y[node] for node in y if node in test_nodes])
        y = torch.from_numpy(y).float()

        indicies = set(enumerate(nodes))
        train_mask = []
        valid_mask = []
        validation_nodes = tmp[self.ont]['valid']
        for i in indicies:
            if i[1] in validation_nodes:
                valid_mask.append(True)
                train_mask.append(False)
            else:
                valid_mask.append(False)
                train_mask.append(True)

        for i, j in zip(train_mask, valid_mask):
            assert i != j and type(i) == type(True) and type(j) == type(True)

        # add node features
        nx.set_node_attributes(G, node_features)

        data = from_networkx(G, group_node_attrs=all)
        # print(data.nodes)
        print(data.generate_ids())
        #print(zip(G.nodes, data.nodes))
        exit()
        data.train_mask = torch.BoolTensor( train_mask)
        data.valid_mask = torch.BoolTensor(valid_mask)
        data.nodes = nodes
        data.y = y
        self.clusters = []
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
