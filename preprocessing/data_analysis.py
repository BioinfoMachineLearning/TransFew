from collections import Counter

import networkx as nx
import obonet
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SwissProt
import networkx
import obonet
import CONSTANTS
from CONSTANTS import exp_evidence_codes


def read_uniprot(in_file, save=False, out_file="uniprot"):
    handle = open(in_file)
    go_graph = obonet.read_obo(open(CONSTANTS.ROOT_DIR + "obo/go-basic.obo", 'r'))
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

        if len(go_terms) > 0:
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


read_uniprot(CONSTANTS.ROOT_DIR + "uniprot/uniprot_sprot.dat", save=True, out_file="uniprot")
exit()

def statistics_go_terms():
    go_graph = obonet.read_obo(open(CONSTANTS.ROOT_DIR + "obo/go-basic.obo", 'r'))
    protein = pd.read_csv("uniprot.csv", sep="\t", index_col=False)

    go_terms = {term: set() for term in go_graph.nodes()}

    for index, row in protein[["ACC", "GO_IDS"]].iterrows():
        if isinstance(row[1], str):
            tmp = row[1].split("\t")
            for term in tmp:
                go_terms[term].add(row[0])

    for ont in CONSTANTS.FUNC_DICT:
        ont_terms = nx.ancestors(go_graph, Constants.FUNC_DICT[ont]).union(set([Constants.FUNC_DICT[ont]]))
        filtered = {key: go_terms[key] for key in go_terms if key in ont_terms}

        tmp = {}
        for i in filtered:
            name = str(int(len(filtered[i]) / 100) * 100) + "_" + str((int(len(filtered[i]) / 100) + 1) * 100)
            print(name)
            exit()
            if name in tmp:
                tmp[name] = tmp[name] + 1
            else:
                tmp[name] = 1

        plt.plot(tmp.keys(), tmp.values(), color='g')
        plt.bar(tmp.keys(), tmp.values(), color='g')
        plt.ylim(0, 100)
        plt.xticks(rotation=90, ha='right')
        plt.show()
    exit()

    count_dict = {}
    for key, value in term_counter.items():
        print(key, value)
        exit()
    #     if value in count_dict:
    #         count_dict[value] +=1
    #     else:
    #         count_dict[value] = 1
    #
    # k = []
    # v = []
    # for key, value in count_dict.items():
    #     k.append(key)
    #     v.append(value)
    #
    #
    # z = Counter(term_counter.values())
    #
    # print(z)
    # exit()
    #
    # # create frequency histogram
    # plt.plot(v, linewidth=2, markersize=12)
    # plt.show()
    # exit()

    # keys = []
    # vals = []
    #
    # for key, value in go_terms.items():
    #     keys.append(key)
    #     vals.append(value)
    #
    # plt.bar(range(len(keys)), vals, tick_label=keys)
    # plt.show()
    #
    # # compute distribution
    #
    # #     try:
    # #         print(row[1])
    # #         go_terms[row[1]].add(row[0])
    # #     except KeyError:
    # #         go_terms[row[1]] = set([row[0],])
    # #
    # # print(go_terms)
    # exit()

    # for node, data in go_graph.nodes(data=True):
    #     # print(node, data['namespace'], len(nx.descendants(go_graph, node)), len(nx.ancestors(go_graph, node)))
    #     a = nx.descendants(go_graph, node).union(set([node]))
    #     # print(node, assert len(a), len(b))

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
