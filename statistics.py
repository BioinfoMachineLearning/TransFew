from collections import Counter

import networkx as nx
import numpy as np
import obonet
import pandas as pd
from matplotlib import pyplot as plt

import CONSTANTS
from Classes.Diamond import Diamond


#
def diamond_graph_distribution():
    kwargs = {
        'fasta_file': "data/Fasta/id.fasta"
    }
    diamond = Diamond("data/{}".format("Diamond"), **kwargs)
    G = diamond.get_graph()

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    daya = np.unique(degree_sequence, return_counts=True)

    axs[0].bar(daya[0], daya[1], label="Label here")
    axs[0].set_title("Degree histogram")
    axs[0].set_xlabel("Degree")
    axs[0].set_ylabel("# of Nodes")

    axs[1].bar(daya[0], daya[1], label="Label here")
    axs[1].set_title("Degree histogram 0 - 10")
    axs[1].set_ylim([0, 10])
    axs[1].set_xlabel("Degree")
    axs[1].set_ylabel("# of Nodes")

    axs[2].bar(daya[0], daya[1], label="Label here")
    axs[2].set_title("Degree histogram 1000 - 6000")
    axs[2].set_ylim([1000, 6000])
    axs[2].set_xlabel("Degree")
    axs[2].set_ylabel("# of Nodes")

    plt.suptitle("Distribution of nodes and degree for diamond graph")
    fig.tight_layout()
    plt.savefig("plots/diamond_distribution.png")


def statistics_go_terms():
    go_graph = obonet.read_obo(open("go-basic.obo", 'r'))
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

        # val 0 == proteins; val 1 == ancs
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



def go_term_distribution():
    go_graph = obonet.read_obo(open(CONSTANTS.ROOT_DIR + "obo/go-basic.obo", 'r'))
    protein = pd.read_csv(CONSTANTS.ROOT_DIR + "training_data.csv", sep="\t", index_col=False)
    go_terms = {term: set() for term in go_graph.nodes()}
    for index, row in protein[["ACC", "GO_IDS"]].iterrows():
        if isinstance(row[1], str):
            tmp = row[1].split("\t")
            for term in tmp:
                go_terms[term].add(row[0])


def go_protein_length_distribution():
    protein = pd.read_csv(CONSTANTS.ROOT_DIR + "training_data.csv", sep="\t", index_col=False)
    lengths = Counter(protein["SEQUENCE LENGTH"].tolist())

    lengths = sorted(lengths.items(), key=lambda item: item[0], reverse=True)
    x, y = list(zip(*lengths))

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.scatter(x, y, label="Label here")
    axs.set_title("Protein length distribution")
    axs.set_xlabel("# of proteins")
    axs.set_ylabel("Length of protein")

    plt.show()
    fig.tight_layout()
    plt.savefig("plots/diamond_distribution.png")


def stats_on_clusters(in_file):
    """
    Find the min and max proteins counts in cluster
    """
    file = open(in_file)
    lines = [line.strip("\n").split("\t") for line in file.readlines() if line.strip()]
    file.close()

    print(len(min(lines, key=len)), len(max(lines, key=len)))

# diamond_graph_distribution()
# go_term_distribution()
# go_protein_length_distribution()
# stats_on_clusters(in_file)
