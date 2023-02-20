import numpy as np
from matplotlib import pyplot as plt

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


# diamond_graph_distribution()