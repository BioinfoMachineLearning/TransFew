import os.path
import pickle
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import CONSTANTS
from Utils import is_file, create_directory


class Diamond:
    """
    Class to handle interpro data
    """

    def __init__(self, dir, **kwargs):
        self.graph = None
        self.dir = dir
        create_directory(self.dir)
        self.fasta = kwargs.get('fasta_file', CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta")
        self.dbase = kwargs.get('dbase', CONSTANTS.ROOT_DIR + "diamond/database")
        self.output = kwargs.get('output', CONSTANTS.ROOT_DIR + "diamond/output.tsv")
        self.path = kwargs.get('output', CONSTANTS.ROOT_DIR + "diamond/graph.adjlist")
        self.ontology = kwargs.get('ontology', "cc")

    def create_diamond_dbase(self):
        print("Creating Diamond Database")
        if is_file(self.fasta):
            CMD = "diamond makedb --in {} -d {}" \
                .format(self.fasta, self.dbase)
            subprocess.call(CMD, shell=True)
        else:
            print("Fasta file not found.")
            exit()

    def query_diamond(self):
        print("Querying Diamond Database")
        CMD = "diamond blastp -q {} -d {} -o {} --very-sensitive" \
            .format(self.fasta, self.dbase, self.output)
        if is_file(self.dbase + ".dmnd"):
            subprocess.call(CMD, shell=True)
        else:
            print("Database not found. Creating database")
            self.create_diamond_dbase()
            print("Querying Diamond Database")
            subprocess.call(CMD, shell=True)

    def create_diamond_graph(self):
        print("Creating Diamond Graph")
        if is_file(self.output):
            self.graph = nx.Graph()
        else:
            self.query_diamond()
            self.graph = nx.Graph()

        adj_list = self.read_file()
        self.graph = nx.Graph(adj_list)

        # Add nodes lost from diamond sequence alignment
        nodes = set(self.graph.nodes)
        protein = set(pd.read_csv(CONSTANTS.ROOT_DIR + "training.csv", sep="\t", index_col=False)['ACC'].tolist())

        left_off = protein.difference(nodes)
        self.graph.add_nodes_from(left_off)
        nx.write_gml(self.graph, self.path)

    def read_file(self):
        scores = open(self.output, 'r')
        lines = []
        for line in scores.readlines():
            tmp = line.split("\t")
            src, des, wgt = tmp[0], tmp[1], float(tmp[2]) / 100
            if src != des:
                lines.append((src, des, {"weight": wgt}))
        return lines

    def get_graph(self):
        if is_file(self.path):
            self.graph = nx.read_gml(self.path)
        else:
            self.create_diamond_graph()

        return self.graph
