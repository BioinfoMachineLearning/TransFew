import random
from collections import Counter

import networkx as nx
import numpy as np
import obonet
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SwissProt
import networkx
import obonet
import CONSTANTS
from CONSTANTS import exp_evidence_codes
from preprocessing.utils import pickle_save, pickle_load


def read_uniprot(in_file, save=False, out_file="uniprot"):
    handle = open(in_file)
    go_graph = obonet.read_obo(open(CONSTANTS.ROOT_DIR + "obo/go-basic.obo", 'r'))
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
                if evidence in exp_evidence_codes:
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

    df = df.loc[(df["SEQUENCE LENGTH"] > 50) & (df["SEQUENCE LENGTH"] <= 5120)]
    print(df)

    # if save:
    #     df.to_csv('{}.csv'.format(out_file), sep='\t', index=False)
    # else:
    #     return df


# read_uniprot(CONSTANTS.ROOT_DIR + "uniprot/uniprot_sprot.dat", save=True, out_file=CONSTANTS.ROOT_DIR +
# "training_data")


# read_uniprot(CONSTANTS.ROOT_DIR + "uniprot/uniprot_sprot.dat", save=True, out_file="uniprot")
#
# x = pd.read_csv(CONSTANTS.ROOT_DIR + '{}.csv'.format("training"), sep='\t')
# print(x)


def generate_labels():
    go_graph = obonet.read_obo(open(CONSTANTS.ROOT_DIR + "obo/go-basic.obo", 'r'))
    protein = pd.read_csv(CONSTANTS.ROOT_DIR + "training.csv", sep="\t", index_col=False)

    go_terms = {term: set() for term in go_graph.nodes()}

    final_terms = {}

    for index, row in protein[["ACC", "GO_IDS"]].iterrows():
        if isinstance(row[1], str):
            tmp = row[1].split("\t")
            for term in tmp:
                go_terms[term].add(row[0])

    for ont in CONSTANTS.FUNC_DICT:
        ont_terms = nx.ancestors(go_graph, CONSTANTS.FUNC_DICT[ont]).union(set([CONSTANTS.FUNC_DICT[ont]]))

        filtered = {key: (len(go_terms[key]), len(nx.ancestors(go_graph, key).union(set([key]))))
                    for key in go_terms if key in ont_terms and len(go_terms[key]) > 0}

        filtered = {key: value for key, value in filtered.items()
                    if value[1] < 100 and 30 < value[0] < 500}

        terms = sorted(filtered.keys())

        final_terms[ont] = terms

    # labels for proteins
    labels = {}
    train_proteins = {}
    for ont in CONSTANTS.FUNC_DICT:
        prot = {}
        train_proteins[ont] = set()
        curr_ont = final_terms[ont]
        for index, row in protein[["ACC", "GO_IDS"]].iterrows():
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


generate_labels()


