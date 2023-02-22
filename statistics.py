from collections import Counter

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


# diamond_graph_distribution()
go_protein_length_distribution()