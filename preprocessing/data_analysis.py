from collections import Counter

import networkx as nx
import obonet
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SwissProt
import networkx
import obonet
import Constants
from Constants import exp_evidence_codes


def read_uniprot(in_file, save=False, out_file="uniprot"):
    handle = open(in_file)
    go_graph = obonet.read_obo(open(Constants.ROOT + "obo/go-basic.obo", 'r'))
    all = [["ACC", "ID", "GO_IDS", "EVIDENCES", "ORGANISM", "TAXONOMY", "DATA CLASS",
            "CREATED", "SEQUENCE UPDATE", "ANNOTATION UPDATE", "SEQUENCE"]]
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

        go_terms = '\t'.join(map(str, go_terms))
        evidences = '\t'.join(map(str, evidences))

        all.append([primary_accession, entry_name, go_terms, evidences,
                    organism, taxonomy, data_class, created,
                    sequence_update, annotation_update, sequence])

    df = pd.DataFrame(all[1:], columns=all[0])

    if save:
        df.to_csv('{}.csv'.format(out_file), sep='\t', index=False)
    else:
        return df


# read_uniprot(Constants.ROOT + "uniprot/uniprot_sprot.dat", save=True, out_file="uniprot")


def statistics_go_terms():
    go_graph = obonet.read_obo(open("go-basic.obo", 'r'))
    protein = pd.read_csv("uniprot.csv", sep="\t", index_col=False)

    go_terms = {term: set() for term in go_graph.nodes()}

    for index, row in protein[["ACC", "GO_IDS"]].iterrows():
        if isinstance(row[1], str):
            tmp = row[1].split("\t")
            for term in tmp:
                go_terms[term].add(row[0])

    for ont in Constants.FUNC_DICT:
        ont_terms = nx.ancestors(go_graph, Constants.FUNC_DICT[ont]).union(set([Constants.FUNC_DICT[ont]]))
        filtered = {key: go_terms[key] for key in go_terms if key in ont_terms}

        filtered = {key: (len(value), len(nx.ancestors(go_graph, key).union(set([key])))) for key, value in
                    filtered.items() if len(value) > 0}

        _sorted = dict(sorted(filtered.items(), key=lambda item: item[1]))

        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        # all data
        x = list(zip(*filtered.values()))
        x, y = x[0], x[1]
        axs[0].scatter(x, y, label=len(filtered))
        axs[0].set_title("Distribution of Ancestors")
        axs[0].set_xlabel("# Proteins")
        axs[0].set_ylabel("Descendants")
        axs[0].legend()

        # first filter < 1000 at least 40 proteins and less than 1000 descendants
        _filtered = {key: value for key, value in filtered.items()
                     if value[1] < 100 and value[0] > 30 and value[0] < 500}
        x = list(zip(*_filtered.values()))
        x, y = x[0], x[1]
        axs[1].scatter(x, y, label=len(_filtered), c="blue")
        axs[1].set_title("Distribution of Ancestors")
        axs[1].set_xlabel("# Proteins")
        axs[1].set_ylabel("Descendants")
        axs[1].legend()

        # removed
        removed = {key: value for key, value in filtered.items()
                   if not key in _filtered}
        x = list(zip(*removed.values()))
        x, y = x[0], x[1]
        axs[2].scatter(x, y, label=len(removed), c="red")
        axs[2].set_title("Distribution of Ancestors")
        axs[2].set_xlabel("# Proteins")
        axs[2].set_ylabel("Descendants")
        axs[2].legend()

        # can be retrieved
        _x = [(i, len(nx.ancestors(go_graph, i).union(set([i])).intersection(set(_filtered.keys())))) for i in removed]
        _x = [i[1] for i in _x]
        _x = Counter(_x)

        axs[3].bar(_x.keys(), _x.values(), align='center', label="0: {}, 1: {}, 2: {}\n"
                                                                 "3: {}, 4: {}, 5: {}".format(
            _x[0], _x[1], _x[2], _x[3], _x[4], _x[5]))
        axs[3].set_ylim([0, 12])
        axs[3].set_title("Ancestors in training")
        axs[3].set_xlabel("# Ancestors in training")
        axs[3].set_ylabel("# of go terms")
        axs[3].legend()

        plt.suptitle("Distribution of nodes and degree for diamond graph -- {}".format(len(ont_terms)))
        fig.tight_layout()
        plt.show()

        exit()

        plt.savefig("plots/diamond_distribution.png")


statistics_go_terms()

exit()


def statistics_proteins():
    protein = pd.read_csv("uniprot", sep="\t", index_col=False)

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    print(protein)
    print(protein.shape)

    # go_graph = obonet.read_obo(open(Constants.ROOT + "obo/go-basic.obo", 'r'))
    # go_rels = Ontology(Constants.ROOT + "obo/go-basic.obo", with_rels=True)

    # for node, data in go_graph.nodes(data=True):
    #     # print(node, data['namespace'], len(nx.descendants(go_graph, node)), len(nx.ancestors(go_graph, node)))
    #     a = nx.descendants(go_graph, node).union(set([node]))
    #     b = go_rels.get_anchestors(node)
    #     print(node, len(a), len(b))

    # go_terms = {term: set() for term in go_graph.nodes()}
    #
    # for index, row in protein[["ACC", "GO_ID"]].iterrows():
    #     try:
    #         print(row[1])
    #         go_terms[row[1]].add(row[0])
    #     except KeyError:
    #         go_terms[row[1]] = set([row[0],])
    #
    # print(go_terms)


statistics_go_terms()

exit()

go_rels = Ontology(Constants.ROOT + "obo/go-basic.obo", with_rels=True)
# go_set = go_rels.get_namespace_terms(Constants.NAMESPACES["cc"])

go_set = go_rels.get_anchestors("GO:0009420")

# print(go_set)
