import os.path as osp

import networkx as nx
import torch
from torch_geometric.data import Dataset, download_url
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from Utils import is_file
import Constants
from Classes.Diamond import Diamond


class DiamondDataset(InMemoryDataset):

    def __init__(self, root: str = "", name: str = "", split: str = "random",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.name = "Diamond"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        #        assert self.split in ['random', 'preset']

        if split == 'preset':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])
        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.valid_mask.fill_(False)
            data.valid_mask[remaining[:num_val]] = True

            # data.test_mask.fill_(False)
            # data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        kwargs = {
            'fasta_file': "data/Fasta/id.fasta"
        }
        diamond = Diamond("data/{}".format("Diamond"), **kwargs)
        G = diamond.get_graph()

        nodes = G.nodes
        node_features = {}
        layer = 36
        tmp = torch.load(Constants.ROOT + "data/Embeddings/output/{}.pt".format('A5A615'))
        for node in nodes:
            if is_file(Constants.ROOT + "data/Embeddings/output/{}.pt".format(node)):
                _x = torch.load(Constants.ROOT + "data/Embeddings/output/{}.pt".format(node))
                node_features[node] = {'{}'.format(layer): _x['mean_representations'][layer].tolist(), 1:2}
            else:
                node_features[node] = {'{}'.format(layer): tmp['mean_representations'][layer].tolist(), 1:2}

        # add node features
        nx.set_node_attributes(G, node_features)


        data = from_networkx(G, group_node_attrs=all, group_edge_attrs=all)
        print(data)
        exit()

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        print(data)

        exit()
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
