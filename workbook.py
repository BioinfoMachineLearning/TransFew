# Notebook to do surplus work
import os
import shutil

import pandas as pd
from Bio import SeqIO

import CONSTANTS
from Classes.Fasta import Fasta
from Utils import count_proteins, create_seqrecord
from preprocessing.utils import pickle_save


def sup_1():
    # Used this to find proteins with sequence length > 1022 to cut and concatenate for esm2_t48
    df = pd.read_csv(CONSTANTS.ROOT_DIR + "training.csv", sep="\t")
    df = df.loc[(df["SEQUENCE LENGTH"] > 1022)]

    proteins = set(df['ACC'].to_list())

    fasta_path = CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta"
    embeddings = Fasta(fasta_path)
    embeddings.subset_from_fasta(CONSTANTS.ROOT_DIR + "uniprot/uniprot_gt_1022.fasta")


# sup_1()


def divide_4_interpro():
    seqs = []
    input_seq_iterator = SeqIO.parse("/home/fbqc9/PycharmProjects/ipscan/uniprot_fasta.fasta", "fasta")
    for pos, record in enumerate(input_seq_iterator):
        seqs.append(create_seqrecord(id=record.id, seq=str(record.seq)))
        if pos % 10000 == 0 and pos != 0:
            SeqIO.write(seqs, CONSTANTS.ROOT_DIR + "uniprot/{}_{}.fasta".format("uniprot_fasta", pos), "fasta")
            seqs = []

    SeqIO.write(seqs, CONSTANTS.ROOT_DIR + "uniprot/{}_{}.fasta".format("uniprot_fasta", pos), "fasta")

# divide_4_interpro()

# print(count_proteins("/home/fbqc9/PycharmProjects/ipscan/uniprot_fasta.fasta"))

print(count_proteins(CONSTANTS.ROOT_DIR + "uniprot/{}_{}.fasta".format("uniprot_fasta", 79220)))