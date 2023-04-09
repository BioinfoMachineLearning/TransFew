# Notebook to do surplus work
import os
import shutil

import pandas as pd

import CONSTANTS
from Classes.Fasta import Fasta
from Utils import count_proteins
from preprocessing.utils import pickle_save


def sup_1():
    # Used this to find proteins with sequence length > 1022 to cut and concatenate for esm2_t48
    df = pd.read_csv(CONSTANTS.ROOT_DIR + "training.csv", sep="\t")
    df = df.loc[(df["SEQUENCE LENGTH"] > 1022)]

    proteins = set(df['ACC'].to_list())

    fasta_path = CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta"
    embeddings = Fasta(fasta_path)
    embeddings.subset_from_fasta(CONSTANTS.ROOT_DIR + "uniprot/uniprot_gt_1022.fasta")


sup_1()
