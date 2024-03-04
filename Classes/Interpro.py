import subprocess
import networkx as nx
import pandas as pd
import CONSTANTS
from Utils import is_file, pickle_save, pickle_load


class Interpro:
    '''
        Class to handle interpro data
    '''

    def __init__(self, ont):

        self.lines = None
        self.raw_graph_file = CONSTANTS.ROOT_DIR + "interpro/ParentChildTreeFile.txt"
        self.graph = nx.DiGraph()
        self.remap_keys = {}
        self.ont = ont
        self.data_path = CONSTANTS.ROOT_DIR + "interpro/interpro_data"
        self.ohe_path = CONSTANTS.ROOT_DIR + "interpro/interpro_ohe_{}".format(self.ont)
        self.categories_path = CONSTANTS.ROOT_DIR + "interpro/categories_{}".format(self.ont)
        self.category_count = CONSTANTS.ROOT_DIR + "interpro/category_count_{}".format(self.ont)
        

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
        rels = open(self.raw_graph_file, 'r')
        self.lines = [i.rstrip('\n').split("::") for i in rels.readlines()]


    def get_graph(self):
        self.propagate_graph()
        return self.graph


    def generate_features():
        # generate features from interproscan.
        files = list(range(10000, 80000, 10000)) + [79220]
        # generate from terminal in chunks
        for file in files:
            CMD = "interproscan-5.61-93.0-64-bit/interproscan-5.61-93.0/interproscan.sh -cpu 10 \
                -i uniprot_fasta_{}.fasta -o ./interpro_out_{} -f TSV --goterms".format(file, file)
            subprocess.call(CMD, shell=True)

    @staticmethod
    def merge_chunks(num_files=23):
        infile = CONSTANTS.ROOT_DIR + "interpro/interpro_out_{}"
        # merge chunks
        df = pd.DataFrame()
        for file in range(1, num_files, 1):
            data = pd.read_csv(infile.format(file), sep="\t",
                            names=["Protein accession", "Sequence MD5", "Sequence length", "Analysis",
                                    "Signature accession", "Signature description", "Start location",
                                    "Stop location", "Score", "Status", "Date",
                                    "InterPro annotations", "InterPro annotations description ", "GO annotations"])
            df = pd.concat([df, data], axis=0)
        # passed additional quality checks and is very unlikely to be a false match.
        df = df[['Protein accession', 'InterPro annotations']]
        df = df[df["InterPro annotations"] != "-"]
        df.to_csv(CONSTANTS.ROOT_DIR + "interpro/interpro_filtered.csv", index=False, sep="\t")


    def get_features(self):
        if not is_file(CONSTANTS.ROOT_DIR + "interpro/interpro_filtered.csv"):
            self.merge_chunks(num_files=23)
        data = pd.read_csv(CONSTANTS.ROOT_DIR + "interpro/interpro_filtered.csv", sep="\t")
        return data
    

    def create_interpro_data(self):
        features = self.get_features()
        self.get_graph()
        data = {}

        for line_number, (index, row) in enumerate(features.iterrows()):
            acc = row[0]
            annot = row[1]
            try:
                tmp = nx.descendants(self.graph, annot) | set([annot])
            except nx.exception.NetworkXError:
                tmp = set([annot])

            if acc in data:
                data[acc].update(tmp) 
            else:
                data[acc] = tmp
        pickle_save(data, self.data_path)


    def create_features(self, ont):

        # Convert Interpro to One-Hot
        if not is_file(self.data_path + ".pickle"):
            print(self.data_path + ".pickle")
            self.create_interpro_data()

        categories = set()
        data =  pickle_load(self.data_path)

        train_proteins = pickle_load(CONSTANTS.ROOT_DIR + "train_validation")
        val_proteins = set(train_proteins[self.ont]['validation'])
        train_proteins = set(train_proteins[self.ont]['train'])

        test_proteins = set(pickle_load(CONSTANTS.ROOT_DIR + "test_proteins"))

        found_proteins = set()
        for protein, category in data.items():
            if protein in train_proteins:
                categories.update(category)
                found_proteins.add(protein)
        
        categories = list(categories)
        categories.sort()

        all_proteins = train_proteins.union(val_proteins).union(test_proteins)
        print("training: {}, validation: {}, test: {}, all: {}".format(len(train_proteins), len(val_proteins), len(test_proteins), len(all_proteins)))
        print("training proteins with interpro: {}".format(len(found_proteins)))

        ohe = {}
        category_count = {i: 0 for i in categories}

        for protein, annots in data.items():
            if protein in all_proteins:
                ohe[protein] = []
                for cat in categories:
                    if cat in annots:
                        ohe[protein].append(1)
                        category_count[cat] = category_count[cat] + 1
                    else:
                        ohe[protein].append(0)

        print(len(ohe.keys()))

        pickle_save(categories, self.categories_path)
        pickle_save(ohe, self.ohe_path)
        pickle_save(category_count, self.category_count)
    

    def get_interpro_ohe_data(self):
        if not is_file(self.ohe_path + ".pickle") or not is_file(self.categories_path + ".pickle"):
            print("creating interpro features")
            self.create_features(self.ont)
        return pickle_load(self.ohe_path), pickle_load(self.categories_path), pickle_load(self.category_count)
    


    def get_interpro_test(self):
        self.get_graph()
        # load test interpro
        '''data = pd.read_csv(CONSTANTS.ROOT_DIR + "interpro/test_intepro.out", sep="\t",
                            names=["Protein accession", "Sequence MD5", "Sequence length", "Analysis",
                                    "Signature accession", "Signature description", "Start location",
                                    "Stop location", "Score", "Status", "Date",
                                    "InterPro annotations", "InterPro annotations description ", "GO annotations"])'''
        data = pd.read_csv(CONSTANTS.ROOT_DIR + "interpro/test_intepro.out", sep="\t",
                            names=["Protein accession", "InterPro annotations"])

        annots = {}
        for line_number, (index, row) in enumerate(data.iterrows()):

            acc = row.iloc[0]
            try:
                tmp = nx.descendants(self.graph, row.iloc[1]) | set([row.iloc[1]]) 
            except nx.exception.NetworkXError:
                tmp = set([row.iloc[1]]) 

            if acc in annots:
                annots[acc].update(tmp)
            else:
                annots[acc] = set(tmp)


        categories = pickle_load(self.categories_path)
        ohe = {}

        for protein, annot in annots.items():
            ohe[protein] = []
            for cat in categories:
                if cat in annot:
                    ohe[protein].append(1)
                else:
                    ohe[protein].append(0)

                

        return ohe, categories, 0
        








def create_indicies():
    onts = ['cc', 'mf', 'bp']

    for ont in onts:
        print("Processing {}".format(ont))
        cat_counts = pickle_load(CONSTANTS.ROOT_DIR + "interpro/category_count_{}".format(ont))
        cats = pickle_load(CONSTANTS.ROOT_DIR + "interpro/categories_{}".format(ont))

        indicies = {3:[], 5:[], 10:[], 50:[], 100:[], 250:[], 500:[]}

        for ind in indicies:
            for pos, cat in enumerate(cats):
                if cat_counts[cat] > ind:
                    indicies[ind].append(pos)
        
        pickle_save(indicies, CONSTANTS.ROOT_DIR + "interpro/indicies_{}".format(ont))

# create_indicies()