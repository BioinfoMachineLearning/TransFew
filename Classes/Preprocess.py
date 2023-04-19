import random

import networkx
import networkx as nx
import obonet
import pandas as pd
from Bio import SwissProt

from Classes.Embeddings import Embeddings
from Classes.Templates import Templates
from Graph.DiamondDataset import DiamondDataset
import CONSTANTS
from Utils import is_file
from preprocessing.utils import pickle_save, pickle_load


class Preprocess:
    """
    This class is used to handle all data generations
    """

    def __init__(self, **kwargs):
        # files
        self.go_path = kwargs.get('go_file', CONSTANTS.ROOT_DIR + "obo/go-basic.obo")
        self.raw_fasta = kwargs.get('raw_fasta', CONSTANTS.ROOT_DIR + "uniprot/uniprot_sprot.dat")
        self.training_file = kwargs.get('training_file', CONSTANTS.ROOT_DIR + "training.csv")
        self.train_val_file = kwargs.get('train_val_file', CONSTANTS.ROOT_DIR + "datasets/training_validation")
        self.label_file = kwargs.get('label_file', CONSTANTS.ROOT_DIR + "datasets/labels")
        self.terms_file = kwargs.get('terms_file', CONSTANTS.ROOT_DIR + "datasets/terms")

        # objects
        self.go_graph = obonet.read_obo(open(self.go_path, 'r'))

        self.run()

    @staticmethod
    def read_uniprot(in_file, go_graph, save=False, out_file="uniprot"):
        print("Generating Data from Raw Fasta")
        handle = open(in_file)
        all = [["ACC", "ID", "GO_IDS", "EVIDENCES", "ORGANISM", "TAXONOMY", "DATA CLASS",
                "CREATED", "SEQUENCE UPDATE", "ANNOTATION UPDATE", "SEQUENCE", "SEQUENCE LENGTH"]]
        for record in SwissProt.parse(handle):
            # accessions, annotation_update, comments, created, cross_references, data_class, description
            # entry_name, features, gene_name, host_organism, host_taxonomy_id, keywords, molecule_type
            # organelle,  organism, organism_classification, protein_existence, references, seqinfo
            # sequence, sequence_length, sequence_update, taxonomy_id
            primary_accession = record.accessions[0]
            entry_name = record.entry_name
            cross_refs = record.cross_references
            organism = record.organism
            taxonomy = record.taxonomy_id
            assert len(taxonomy) == 1
            taxonomy = taxonomy[0]
            data_class = record.data_class
            created = record.created[0]
            sequence_update = record.sequence_update[0]
            annotation_update = record.annotation_update[0]
            sequence = record.sequence
            sequence_length = len(record.sequence)
            go_terms = []
            evidences = []
            for ref in cross_refs:
                if ref[0] == "GO":
                    assert len(ref) == 4
                    go_id = ref[1]
                    evidence = ref[3].split(":")[0]
                    if evidence in CONSTANTS.exp_evidence_codes:
                        try:
                            tmp = nx.descendants(go_graph, go_id).union(set([go_id]))
                            go_terms.extend(tmp)
                            evidences.extend([evidence] * len(tmp))
                        except networkx.exception.NetworkXError:
                            pass

            if len(go_terms) > 0:
                go_terms = '\t'.join(map(str, go_terms))
                evidences = '\t'.join(map(str, evidences))

                all.append([primary_accession, entry_name, go_terms, evidences,
                            organism, taxonomy, data_class, created,
                            sequence_update, annotation_update, sequence, sequence_length])

        df = pd.DataFrame(all[1:], columns=all[0])

        # df = df.loc[(df["SEQUENCE LENGTH"] > 50) & (df["SEQUENCE LENGTH"] <= 5120)]

        if save:
            df.to_csv('{}.csv'.format(out_file), sep='\t', index=False)
        else:
            print(df.head(10))
            # return df
        print("Raw processing finished. Only proteins with sequence length between 50 and 5121 are kept.")

    @staticmethod
    def generate_labels(go_graph, training_data):
        print("Filtering and Selecting GO Terms")
        go_terms = {term: set() for term in go_graph.nodes()}
        final_terms = {}

        for index, row in training_data[["ACC", "GO_IDS"]].iterrows():
            if isinstance(row[1], str):
                tmp = row[1].split("\t")
                for term in tmp:
                    go_terms[term].add(row[0])

        for ont in CONSTANTS.FUNC_DICT:
            ont_terms = nx.ancestors(go_graph, CONSTANTS.FUNC_DICT[ont]).union(set([CONSTANTS.FUNC_DICT[ont]]))

            filtered = {key: go_terms[key] for key in go_terms if key in ont_terms}

            filtered = {key: (len(value), len(nx.ancestors(go_graph, key).union(set([key])))) for key, value in
                        filtered.items()}

            filtered = {term: (num_prot, ancs) for term, (num_prot, ancs) in filtered.items() if num_prot > 100 > ancs}
            #
            # print(len(filtered))
            #
            # exit()

            # for i in filtered.items():
            #     print(i)
            # exit()

            # filtered = {key: value for key, value in filtered.items()
            #             if value[1] < 100 and 30 < value[0] < 500}

            terms = sorted(filtered.keys())

            final_terms[ont] = terms

        # labels for proteins
        labels = {}
        train_proteins = {}
        for ont in CONSTANTS.FUNC_DICT:
            prot = {}
            train_proteins[ont] = set()
            curr_ont = final_terms[ont]
            for index, row in training_data[["ACC", "GO_IDS"]].iterrows():
                _protein = row[0]
                tmp_arr = []
                tmp = set(row[1].split("\t"))
                for term in curr_ont:
                    if term in tmp:
                        tmp_arr.append(1)
                    else:
                        tmp_arr.append(0)
                if sum(tmp_arr) > 0:
                    prot[_protein] = tmp_arr
                    train_proteins[ont].add(_protein)
            labels[ont] = prot

        # Train validation indicies
        training_proteins = {}
        for ont in train_proteins:
            training_proteins[ont] = {}
            tot = int(0.15 * len(train_proteins[ont]))
            indicies = random.sample(range(0, len(train_proteins[ont])), tot)

            _all = list(train_proteins[ont])
            _valid = set([_all[i] for i in indicies])
            _train = train_proteins[ont].difference(_valid)

            assert len(_train.intersection(_valid)) == 0
            assert len(_train) + len(_valid) == len(train_proteins[ont])

            training_proteins[ont]['train'] = _train
            training_proteins[ont]['valid'] = _valid

        pickle_save(labels, CONSTANTS.ROOT_DIR + "datasets/labels")
        pickle_save(training_proteins, CONSTANTS.ROOT_DIR + "datasets/training_validation")
        pickle_save(final_terms, CONSTANTS.ROOT_DIR + "datasets/terms")

    @staticmethod
    def generate_diamond():
        for ont in CONSTANTS.ontologies:
            DiamondDataset(ont=ont)

    @staticmethod
    def generate_embedding():
        kwargs = {'fasta': CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta"}
        emb = Embeddings(**kwargs)
        emb.run()

    @staticmethod
    def generate_templates():
        templates = Templates()
        templates.generate_templates()

    def run(self):
        # create training data
        if not is_file(self.training_file):
            self.read_uniprot(self.raw_fasta, self.go_graph, save=True, out_file=self.training_file)
        training_data = pd.read_csv(self.training_file, sep="\t", index_col=False)

        prots = training_data['ACC'].to_list()

        pickle_save(prots, "all_proteins")
        exit()

        # generate labels
        if not is_file("{}.pickle".format(self.train_val_file)) or not is_file(
                "{}.pickle".format(self.label_file)) or not is_file("{}.pickle".format(self.terms_file)):
            self.generate_labels(self.go_graph, training_data, self.train_val_file, self.label_file, self.terms_file)

        training_validation = pickle_load(self.train_val_file)
        labels = pickle_load(self.label_file)
        terms = pickle_load(self.terms_file)

        print("++++++++++++++++++++++ Data Statistics ++++++++++++++++++++++")
        for ont in CONSTANTS.ontologies:
            print("Ontology: {}, # of training proteins {}, # of validation proteins {}, # of total {}". \
                  format(len(terms[ont]), len(training_validation[ont]['train']),
                         len(training_validation[ont]['valid']), len(labels[ont].keys())))
        print("++++++++++++++++++++++ Finished Generating Labels ++++++++++++++++++++++")

        # Generate Diamond
        print("++++++++++++++++++++++ Generating Diamond Graph ++++++++++++++++++++++")
        self.generate_diamond()
        print("++++++++++++++++++++++ Diamond Graph Generation Finished ++++++++++++++++++++++")

        # # Generate Embedding
        # print("++++++++++++++++++++++ Generating Embedding ++++++++++++++++++++++")
        # self.generate_embedding()
        # print("++++++++++++++++++++++ Embedding Generation Finished ++++++++++++++++++++++")
        #
        # # Generate Templates
        # print("++++++++++++++++++++++ Generating Embedding ++++++++++++++++++++++")
        # self.generate_templates()
        # print("++++++++++++++++++++++ Embedding Generation Finished ++++++++++++++++++++++")

        # per residue etc


x = Preprocess()
