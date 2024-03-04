import csv
import os
import random
import subprocess
import networkx
import networkx as nx
import obonet
import pandas as pd
from Bio import SwissProt
import CONSTANTS
from Classes.Fasta import Fasta
from Utils import count_proteins, is_file, read_cafa5_scores, readlines_cluster
from Utils import pickle_save, pickle_load
from Bio import SeqIO
from operator import itemgetter
# from Classes.Embeddings import Embeddings
# from Classes.Templates import Templates
# from Graph.DiamondDataset import DiamondDataset 


class Preprocess:
    """
    This class is used to handle all data generations
    """

    def __init__(self, **kwargs):
        # files
        self.go_path = kwargs.get('go_file', CONSTANTS.go_graph_path)
        self.raw_fasta = kwargs.get('raw_fasta', CONSTANTS.ROOT_DIR + "uniprot/train_sequences.fasta")
        self.training_file = kwargs.get('training_file', CONSTANTS.ROOT_DIR + "training.csv")
        self.train_val_file = kwargs.get('train_val_file', CONSTANTS.ROOT_DIR + "training_validation")
        self.label_file = kwargs.get('label_file', CONSTANTS.ROOT_DIR + "labels")
        self.groundtruth = kwargs.get('groundtruth', CONSTANTS.ROOT_DIR + "groundtruth")
        self.terms_file = kwargs.get('terms_file', CONSTANTS.ROOT_DIR + "terms")
        self.all_proteins = kwargs.get('all_proteins', CONSTANTS.ROOT_DIR + "all_proteins")

        # objects
        self.go_graph = obonet.read_obo(open(self.go_path, 'r'))

        accepted_edges = set()
        unaccepted_edges = set()

        for edge in self.go_graph.edges:
            if edge[2] == 'is_a' or edge[2] == 'part_of':
                accepted_edges.add(edge)
            else:
                unaccepted_edges.add(edge)
        self.go_graph.remove_edges_from(unaccepted_edges)


        self.run()

    
    @staticmethod
    # Convert mmseq cluster to one line format
    def one_line_format(input_file, dir):
        data = {}
        with open(input_file) as file:
            lines = file.read().splitlines()
            for line in lines:
                x = line.split("\t")
                if x[0] in data:
                    data[x[0]].append(x[1])
                else:
                    data[x[0]] = list([x[1]])
        result = [data[i] for i in data]
        with open(dir + "/final_clusters.csv", "w") as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerows(result)


    def cluster_sequence(self, seq_id, proteins, ontology):
        wd = CONSTANTS.ROOT_DIR + "{}/mmseq_{}".format(ontology, seq_id)
        if not os.path.exists(wd):
            os.makedirs(wd)

        x = Fasta(self.raw_fasta)
        x = x.fasta_to_list(_filter=proteins)
        SeqIO.write(x, wd+"/fasta", "fasta")

        print("Number of proteins to cluster {}".format(count_proteins(wd+"/fasta")))
            
        CMD = "mmseqs createdb {} {} ; " \
                  "mmseqs cluster {} {} tmp --min-seq-id {};" \
                  "mmseqs createtsv {} {} {} {}.tsv" \
                  "".format(wd+"/fasta", "targetDB", "targetDB", "outputClu", 
                            seq_id, "targetDB", "targetDB",
                            "outputClu", "outputClu")
        subprocess.call(CMD, shell=True, cwd="{}".format(wd))
        self.one_line_format(wd + "/outputClu.tsv", wd)


    @staticmethod
    def generate_labels(go_graph, terms_file, out_file):
        # groundtruth = {}
        # for index, row in terms_file.iterrows():
        #     acc, term = row[0], row[1]
        #     tmp = nx.descendants(go_graph, term).union(set([term]))
        #     if acc in groundtruth:
        #         groundtruth[acc].update(tmp)
        #     else:
        #         groundtruth[acc] = tmp

        # groundtruth1 = groundtruth.copy()
        # print(len(groundtruth))

        groundtruth = {}
        terms = pd.read_csv(CONSTANTS.ROOT_DIR + "cafa5/Train/train_terms.tsv", sep='\t')
        for index, row in terms.iterrows():
            acc, term, aspect = row[0], row[1], row[2]
            if acc in groundtruth:
                groundtruth[acc].add(term)
            else:
                groundtruth[acc] = set([term,])

        print(len(groundtruth))

        # for key in groundtruth:
        #     assert(groundtruth[key] == groundtruth1[key])
        pickle_save(groundtruth, out_file)


        # res = pickle_load("res")
        # print("here")
        # for i in res:
        #     assert len(res[i].difference(groundtruth[i])) == 0
        # exit()


    def generate_train_validation(self, groundtruth):
        print("Filtering and Selecting GO Terms")
        go_graph = self.go_graph

        GO_weights_file = read_cafa5_scores(CONSTANTS.ROOT_DIR + "/cafa5/IA.txt")
        GO_weights = {} 
        for term in GO_weights_file:
            key, value = term.strip().split("\t")
            GO_weights[key] = value


        for ont in CONSTANTS.FUNC_DICT:
            print("Ontology {}".format(ont))
            ont_terms = nx.ancestors(go_graph, CONSTANTS.FUNC_DICT[ont]).union(set([CONSTANTS.FUNC_DICT[ont]]))


            # filter proteins with terms
            # filter proteins and terms related to the current ont
            filtered = {}
            for key in groundtruth:
                tmp = groundtruth[key].intersection(ont_terms)
                if len(tmp) > 0:
                    filtered[key] = tmp
            print("Number of proteins with terms {}".format(len(filtered)))


            # Get statistics on proteins and number of terms.
            prot_num_terms = {key: len(value) for key, value in filtered.items()}
            df = pd.DataFrame.from_dict(prot_num_terms, orient='index').reset_index()
            # df = df.rename(columns={'index': 'Protein', 0: '# of terms'})
            # df.sort_values('# of terms', inplace=True, ascending=False)
            # print(df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
            # cc --> min:2 max: 102.


            # Go terms and number of associated proteins
            # Each go term and number of proteins

            # go2prot = {goterm:set() for goterm in ont_terms}
            # for key in filtered:
            #     tms = filtered[key]
            #     for _tms in tms:
            #         go2prot[_tms].add(key)
            go2prot = {}
            for key in filtered:
                tms = filtered[key]
                for _tms in tms:
                    if _tms in go2prot:
                        go2prot[_tms].add(key)
                    else:
                        go2prot[_tms] = set([key,])
            pickle_save(go2prot, CONSTANTS.ROOT_DIR + "{}/go2prot".format(ont))
            
            sorted_terms = list(go2prot.keys())
            sorted_terms.sort()
            pickle_save(sorted_terms, CONSTANTS.ROOT_DIR+"/{}/sorted_terms".format(ont))
            
            ont_order = {term: pos for pos, term in enumerate(sorted_terms)}

            go_terms = {key: (len(value), 
                              len(nx.ancestors(go_graph, key).union(set([key]))), 
                              len(nx.descendants(go_graph, key).union(set([key]))), 
                              len(nx.shortest_path(go_graph, source=key, 
                                                   target=CONSTANTS.FUNC_DICT[ont])),
                              float(GO_weights[key]), ont_order[key]) 
                              for key, value in go2prot.items()}
            
            pickle_save(go_terms, CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
            

        #     # stats on the GO TERMS.
        #     df = pd.DataFrame.from_dict(go_terms, orient='index').reset_index()
        #     df = df.rename(columns={'index': 'GO Terms', 
        #                     0: '# of proteins',
        #                     1: 'Anscestors', 
        #                     2: 'Descendants', 
        #                     3: 'Depth',
        #                     4: 'Score',
        #                     5: 'Position'})

        #     df.sort_values('# of proteins', inplace=True, ascending=False)
        #     print(df.describe(percentiles=[.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90]))
        #     # df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).to_excel(writer, sheet_name='{}'.format(ont))

    
            
            # Generate labels
            print("Generating labels")
            labels = {}
            for prot in filtered:
                tmp = []
                prot_terms = set(filtered[prot])
                for t in sorted_terms:
                    if t in prot_terms:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                labels[prot] = tmp
            pickle_save(labels, CONSTANTS.ROOT_DIR+"/{}/labels".format(ont))


        # #     print("Clustering")
        # #     # # Sequence Identity
        # #     # # https://www.scitepress.org/Papers/2009/13792/13792.pdf
        # #     # # x% sequence identity removal
        # #     seq_id = 0.6
        # #     cluster_file = CONSTANTS.ROOT_DIR + "{}/mmseq_{}/final_clusters.csv".format(ont, seq_id)
        # #     # cluster proteins in this ontology
        # #     if not is_file(cluster_file):
        # #         self.cluster_sequence(seq_id=seq_id, proteins=filtered.keys(), ontology=ont)
        # #     clusters = readlines_cluster(cluster_file)


            ont_proteins = list(filtered.keys())
            indicies = list(range(0, len(ont_proteins)))
            random.shuffle(indicies)

            k = int(len(indicies) * 0.9)
            
            train_indicies, validation_indicies = indicies[:k], indicies[k:]
            assert len(train_indicies) + len(validation_indicies) == len(indicies)
            assert len(train_indicies) > len(validation_indicies)

            pickle_save(ont_proteins, CONSTANTS.ROOT_DIR + "{}/all_proteins".format(ont))
            pickle_save(train_indicies, CONSTANTS.ROOT_DIR + "{}/train_indicies".format(ont))
            pickle_save(validation_indicies, CONSTANTS.ROOT_DIR + "{}/validation_indicies".format(ont))

            train = set([ont_proteins[i] for i in train_indicies])
            validation = set([ont_proteins[i] for i in validation_indicies])

            print(len(train), len(validation))

            assert len(train) + len(validation) == len(filtered)

            pickle_save(train, CONSTANTS.ROOT_DIR + "{}/train_proteins".format(ont))
            pickle_save(validation, CONSTANTS.ROOT_DIR + "{}/validation_proteins".format(ont))


    def generate_count_indicies(self):

        thresholds = [0, 30, 50, 100, 250]
        # Added 5 to bp

        max_count = {'mf' : 70000, 'bp': 70000, 'cc': 92000}

        for ont in CONSTANTS.FUNC_DICT:
            data = pickle_load(CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
            indicies = {}
            for threshold in thresholds:
                _tmp = []
                for i in data:
                    # of proteins 'Anscestors' 'Descendants' 'Depth' 'Score' 'Position'
                    if data[i][0] > threshold and data[i][0] < max_count[ont]:
                        _tmp.append(data[i][5])
                indicies[threshold] = _tmp

            indicies['all'] = [data[i][5] for i in data]

            pickle_save(indicies, CONSTANTS.ROOT_DIR + "{}/term_indicies".format(ont))
            

# min_max_map = {
#     'cc': (80000, 30),
#     'mf': (60000 , 30),
#     'bp': (70000, 100)
# }

    def run(self):
        if not is_file("{}.pickle".format(self.all_proteins)):
            print("here")
            cafa5 = Fasta(self.raw_fasta)
            prots = [seq.id for seq in cafa5.fasta_to_list()]
            pickle_save(prots, "{}.pickle".format(self.all_proteins))
        else:
            prots = pickle_load(self.all_proteins)


        if not is_file("{}.pickle".format(self.groundtruth)):
            terms = pd.read_csv(CONSTANTS.ROOT_DIR + "cafa5/Train/train_terms.tsv", sep='\t')  
            self.generate_labels(self.go_graph, terms, self.groundtruth)          
        else:
            print("Labels already generated")
            groundtruth = pickle_load(self.groundtruth)

        self.generate_train_validation(groundtruth)
        self.generate_count_indicies()


x = Preprocess()


x = pickle_load(CONSTANTS.ROOT_DIR + "{}/term_indicies".format('bp'))

for i in x:
    print(i, len(x[i]))
