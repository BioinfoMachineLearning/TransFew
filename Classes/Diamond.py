import os.path
import pickle
import subprocess
import networkx as nx
import numpy as np
import pandas as pd
import torch
import CONSTANTS
import os.path as osp
import torch_geometric.data as pygdata
from Utils import is_file, pickle_load, pickle_save

class Diamond:
    """
    Class to handle Diamond data
    """

    def __init__(self, session='train', **kwargs):
        self.session = session

        if session == 'train': 
            self.fasta = kwargs.get('fasta_file', CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta")
            self.dbase = kwargs.get('dbase', CONSTANTS.ROOT_DIR + "diamond/database")
            self.query = kwargs.get('query', CONSTANTS.ROOT_DIR + "diamond/query.tsv")
            self.adjlist = kwargs.get('output', CONSTANTS.ROOT_DIR + "diamond/graph.list")
        elif session == 'test':
            self.fasta = kwargs.get('fasta_file', CONSTANTS.ROOT_DIR + "cafa5/Test_Target/testsuperset.fasta")
            self.dbase = kwargs.get('dbase', CONSTANTS.ROOT_DIR + "diamond/database")
            self.query = kwargs.get('query', CONSTANTS.ROOT_DIR + "diamond/test_query.tsv")
            self.adjlist = kwargs.get('output', CONSTANTS.ROOT_DIR + "diamond/test_graph.list")
    
    def create_diamond_dbase(self):
        print("Creating Diamond Database")
        if is_file(self.fasta):
            CMD = "diamond makedb --in {} -d {}" \
                .format(self.fasta, self.dbase)
            subprocess.call(CMD, shell=True)
        else:
            print("Fasta file not found.")
            exit()

    def query_diamond(self):
        print("Querying Diamond Database")
        CMD = "diamond blastp -q {} -d {} -o {} --sensitive" \
            .format(self.fasta, self.dbase, self.query)
        if is_file(self.dbase + ".dmnd"):
            subprocess.call(CMD, shell=True)
        else:
            print("Database not found. Creating database")
            self.create_diamond_dbase()
            print("Querying Diamond Database")
            subprocess.call(CMD, shell=True)

    def create_diamond_graph(self):
        if not is_file(self.query):
            self.query_diamond()
        self.read_file()

    def read_file(self):
        scores = open(self.query, 'r')
        res = {}
        for line in scores.readlines():
            tmp = line.split("\t")
            src, des, wgt = tmp[0], tmp[1], float(tmp[2]) / 100

            if src not in res:
                res[src] = {}
            res[src][des] = wgt

            if des not in res:
                res[des] = {}
            res[des][src] = wgt

        pickle_save(res, self.adjlist)

    def get_graph(self):

        if not is_file(self.adjlist + ".pickle"):
            self.create_diamond_graph()
        res = pickle_load(self.adjlist)
        return res
    

    def get_graph_addlist(self):

        # load test diamond

        # fasta = "/home/fbqc9/Workspace/DATA/uniprot/test_proteins.fasta"
        # query = CONSTANTS.ROOT_DIR + "diamond/eval_test_query.tsv"

        # CMD = "diamond blastp -q {} -d {} -o {} --sensitive" \
        #     .format(fasta, self.dbase, query)

        # subprocess.call(CMD, shell=True)

        
        query = CONSTANTS.ROOT_DIR + "diamond/eval_test_query.tsv"
        scores = open(query, 'r')
        res = {}
        for line in scores.readlines():
            tmp = line.split("\t")
            src, des, wgt = tmp[0], tmp[1], float(tmp[2]) / 100

            if src not in res:
                res[src] = []
            res[src].append((des, wgt))

            if des not in res:
                res[des] = []
            res[des].append((src, wgt))

        return res


    def create_pytorch_graph(self, ):

        onts = ['cc', 'mf', 'bp']

        for ont in onts:

            labels = pickle_load(CONSTANTS.ROOT_DIR + "{}/labels".format(ont))
            proteins = pickle_load(CONSTANTS.ROOT_DIR + "{}/all_proteins".format(ont))
            indicies = list(range(0, len(proteins)))
            val_indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/validation_indicies".format(ont))
            train_indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/train_indicies".format(ont))


            protein_dic = { prot: pos for pos, prot in enumerate(proteins)}

            embeddings = []
            ys = []
            interactions = self.get_graph()

            rows = []
            columns = []
            weights = []
            for src in interactions:
                for des, score in interactions[src].items():
                    if src == des:
                        pass
                    else:
                        try:
                            _row = protein_dic[src]
                            _col = protein_dic[des]
                            rows.append(_row)
                            columns.append(_col)
                            weights.append(score)
                        except KeyError:
                            pass

            assert len(rows) ==  len(columns) == len(weights)

            rows = np.array(rows, dtype='int64')
            columns = np.array(columns, dtype='int64')
            edges = torch.tensor(np.array([rows, columns]))

            edges_attr = torch.tensor(np.array(weights, dtype='float32'))

            
            for pos, prt in enumerate(proteins):
                print(pos, len(proteins))
                tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(prt))
                esm = tmp['esm2_t48'].x#.squeeze(0)
                embeddings.append(esm)
                
                ys.append(torch.tensor(labels[prt], dtype=torch.float32).view(1, -1))

            embeddings = torch.cat(embeddings, dim=0)
            ys = torch.cat(ys, dim=0)

            x_x = [False] * 92912
            for i in train_indicies:
                x_x[i] = True
            train_indicies = torch.tensor(np.array(x_x))

            x_x = [False] * 92912
            for i in val_indicies:
                x_x[i] = True
            val_indicies = torch.tensor(np.array(x_x))

            graph = pygdata.Data(num_nodes=len(proteins), edge_index=edges, 
                                 x=embeddings, y=ys, edges_attr=edges_attr,
                                 train_mask=torch.tensor(train_indicies), 
                                 val_mask=torch.tensor(val_indicies))
            

            torch.save(graph, osp.join(CONSTANTS.ROOT_DIR + "{}_tformer_data.pt".format(ont)))



        


    

