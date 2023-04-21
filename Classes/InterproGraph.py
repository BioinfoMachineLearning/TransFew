import networkx as nx
import pandas as pd

from Utils import is_file
from preprocessing.utils import pickle_save, pickle_load


class Interpro:
    '''
        Class to handle interpro data
    '''

    def __init__(self, file):
        self.lines = None
        self.file = file
        self.graph = nx.DiGraph()
        self.remap_keys = {}

    def propagate_graph(self):
        self.read_file()

        for node in self.lines:
            # 8 dashes
            if node[0].startswith("--------"):
                l5 = node[0].strip("--------")
                self.graph.add_edges_from([(l4, l5)])
            # 6 dashes
            elif node[0].startswith("------"):
                l4 = node[0].strip("------")
                self.graph.add_edges_from([(l3, l4)])
            # 4 dashes
            elif node[0].startswith("----"):
                l3 = node[0].strip("----")
                self.graph.add_edges_from([(l2, l3)])
            # 2 dashes
            elif node[0].startswith("--"):
                l2 = node[0].strip("--")
                self.graph.add_edges_from([(l1, l2)])
            else:
                l1 = node[0]
                if not self.graph.has_node(l1):
                    self.graph.add_node(l1)

    def read_file(self):
        rels = open(self.file, 'r')
        self.lines = [i.rstrip('\n').split("::") for i in rels.readlines()]

    def convert_uniprot(self):
        if is_file("../data/interpro/uniprot2ipr"):
            self.remap_keys = pickle_load(self.remap_keys, "../data/interpro/uniprot2ipr")
        else:
            with open("../data/interpro/protein2ipr.dat") as file:
                proteins = set(pd.read_csv("../data/uniprot.csv", sep="\t", index_col=False)['ACC'].tolist())
                for line in file:
                    key = line.split("\t")
                    uniprot, intepro, signature = key[0], key[1], key[3]
                    if uniprot in proteins:
                        self.remap_keys[intepro] = (uniprot, intepro, signature)
            pickle_save(self.remap_keys, "../data/interpro/uni2ipr2sig")

    def get_graph(self):
        return self.graph


# _graph = Interpro("../data/interpro/ParentChildTreeFile.txt")
# _graph.propagate_graph()
# graph = _graph.get_graph()
# _graph.convert_uniprot()


