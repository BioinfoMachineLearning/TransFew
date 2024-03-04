import CONSTANTS
import pickle, os
import pandas as pd
import numpy as np
import torch
import os.path as osp
import torch_geometric.data as pygdata
from Utils import is_file, pickle_save
from torch_geometric.data import Data

def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)

class STRING:
    '''
        Class to handle STRING data
    '''

    def __init__(self, session='train'):
        self.string_dbase = CONSTANTS.ROOT_DIR + "STRING/protein.links.v11.5.txt"
        self.mapping_file = CONSTANTS.ROOT_DIR + "STRING/protein.aliases.v11.5.txt"
        self.mapping = {}
        self.mapping_filtered = CONSTANTS.ROOT_DIR + "STRING/filtered"

        ###
        self.string_file = CONSTANTS.ROOT_DIR + "STRING/string.csv"
        self.neighbours_file = CONSTANTS.ROOT_DIR + "STRING/neighbours"
        

    def extract_mapping(self):
        if not is_file(self.mapping_filtered + ".pickle"):
            print("extracting mappings")
            proteins = set(pickle_load(CONSTANTS.ROOT_DIR + "all_proteins_cafa"))
            test_set = set(pickle_load(CONSTANTS.ROOT_DIR + "test_proteins"))
            proteins = proteins.union(test_set)

            print(len(proteins))

            with open(self.mapping_file) as in_file:
                next(in_file)
                for line in in_file:
                    x = line.strip().split("\t")
                    if x[2] == "BLAST_UniProt_AC" and x[1] in proteins: 
                        self.mapping[x[0]] = x[1]
            pickle_save(self.mapping, self.mapping_filtered)
        else:
            print("mapping file exist, loading")
            self.extract_mapping = pickle_load(self.mapping_filtered)
        print("Mapping finished")

    def extract_uniprot(self):
        if not is_file(self.string_file + ".pickle"):
            self.mapping = pickle_load(self.mapping_filtered)
            interactions = [["String protein 1", "String protein 2", "Combined score", "Uniprot protein 1", "Uniprot protein 2"]]
            with open(self.string_dbase) as in_file:
                next(in_file)
                for line in in_file:
                    x = line.strip().split(" ")
                    if x[0] in self.mapping and x[1] in self.mapping:
                        interactions.append([x[0], x[1], int(x[2]), self.mapping[x[0]], self.mapping[x[1]]])

            df = pd.DataFrame(interactions[1:], columns=interactions[0])
            df.to_csv(self.string_file, sep='\t', index=False)


    def get_String(self, confidence=0.7):

        if not is_file(self.string_file):
            print("generating interactions")
            self.extract_mapping()
            self.extract_uniprot()
        
        data = pd.read_csv(self.string_file, sep='\t') 
        data = data[["Uniprot protein 1", "Uniprot protein 2", "Combined score"]]
        data = data[data["Combined score"] > confidence * 1000]
        return data
    

    def get_string_neighbours(self, ontology, confidence=0.7, recreate=False):

        proteins = set(pickle_load(CONSTANTS.ROOT_DIR + "/{}/all_proteins".format(ontology)))

        if recreate == True or not is_file(self.neighbours_file + "_" + ontology + ".pickle"):
            x = self.get_String(confidence=confidence)

            x = x[x['Uniprot protein 1'].isin(proteins) & x['Uniprot protein 2'].isin(proteins)]


            data = {}
            for index, row in x.iterrows():
                p1, p2, prob = row[0], row[1], row[2]
                
                if p1 in data:
                    data[p1].add(p2)
                else:
                    data[p1] = set([p2, ])

                if p2 in data:
                    data[p2].add(p1)
                else:
                    data[p2] = set([p1, ])

            pickle_save(data, self.neighbours_file + "_" + ontology)
        else:
            data = pickle_load(self.neighbours_file + "_" + ontology)

        return data


    def create_pytorch_graph(self, ont):

        proteins = pickle_load(CONSTANTS.ROOT_DIR + "{}/all_proteins".format(ont))

        labels = pickle_load(CONSTANTS.ROOT_DIR + "{}/labels".format(ont))


        x = []
        y = []
        for j, i in enumerate(proteins):
            tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))
            string_data = tmp['string_{}'.format(ont)].x
            x.append(torch.mean(string_data, dim=0).unsqueeze(0))
            y. append(torch.tensor(labels[i], dtype=torch.long).unsqueeze(0))

        
        y = torch.cat(y, dim=0)
        x = torch.cat(x, dim=0)

        protein_dic = { prot: pos for pos, prot in enumerate(proteins)}

        interactions = self.get_string_neighbours(ont)
    

        #rows =  list(protein_dic.values()) # []
        #cols =  list(protein_dic.values()) # []
        rows = []
        cols = []
        for src in interactions:
            for des in interactions[src]:
                if src == des:
                    pass
                else:
                    # sort small first
                    _row = protein_dic[src]
                    _col = protein_dic[des]


                    rows.append(_row)
                    cols.append(_col)

                    rows.append(_col)
                    cols.append(_row)

        assert len(rows) ==  len(cols)


        nodes = np.unique(rows + cols)

        rows = np.array(rows, dtype='int64')
        cols = np.array(cols, dtype='int64')
        edges = torch.tensor(np.array([rows, cols]))

        train_size = int(len(nodes)*0.85)
        val_size = len(nodes) - train_size

        train_set = nodes[0:train_size]
        val_set = nodes[train_size:]


        # assert len(train_set)+len(val_set) == len(nodes) == len(protein_dic)


        # train_mask = torch.zeros(len(nodes),dtype=torch.long)
        # for i in train_set:
        #     train_mask[i] = 1.
            
        # val_mask = torch.zeros(len(nodes),dtype=torch.long)
        # for i in val_set:
        #     val_mask[i] = 1.


        # data = Data(edge_index=edges, train_mask=train_mask, val_mask=val_mask,
        #             key_val=protein_dic, x=x, y=y)

        data = Data(edge_index=edges,  x=x, y=y, key_val=protein_dic)

        
        torch.save(data, osp.join(CONSTANTS.ROOT_DIR + "{}/node2vec.pt".format(ont)))
            
    def get_embeddings(self, ontology):
        path = CONSTANTS.ROOT_DIR + "{}/node2vec.pt".format(ontology)
        if is_file(path):
            pass
        else:
            print(" generating node2vec graph")
            self.create_pytorch_graph(ontology)

        return torch.load(path)



# ROOT = "/home/fbqc9/Workspace/DATA/"
# String = STRING(ROOT+"STRING/protein.links.v11.5.txt", ROOT+"STRING/protein.aliases.v11.5.txt")
# # String.extract_mapping()
# # String.extract_uniprot()
# data = String.get_String()

# data = data[["Uniprot protein 1", "Uniprot protein 2", "Combined score"]]
# print(data["Combined score"].describe().apply(lambda x: format(x, 'f')))