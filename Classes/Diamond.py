import os.path
import subprocess

import networkx as nx

from Utils import is_file, check_directory


class Diamond:
    """
    Class to handle interpro data
    """
    def __init__(self, dir, **kwargs):
        self.graph = None
        self.dir = dir
        check_directory(self.dir)
        self.fasta = kwargs.get('fasta_file', None)
        self.dbase = kwargs.get('dbase', "../data/Diamond/database")
        self.output = kwargs.get('output', "../data/Diamond/output.tsv")

    def create_diamond_dbase(self):
        print("Creating Diamond Database")
        if is_file(self.fasta):
            CMD = "D:/Workspace/python-3/TFUN/diamond.exe makedb --in {} -d {}" \
                .format(self.fasta, self.dbase)
            subprocess.call(CMD, shell=True)
        else:
            print("Fasta file not found.")
            exit()

    def query_diamond(self):
        print("Querying Diamond Database")
        CMD = "D:/Workspace/python-3/TFUN/diamond.exe blastp -q {} -d {} -o {} --very-sensitive" \
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

    def read_file(self):
        scores = open(self.output, 'r')
        lines = []
        for line in scores.readlines():
            tmp = line.strip("\n")
            src, des, wgt = tmp[0], tmp[1], tmp[2]/100
            print(src, des, wgt)
            exit()
            lines.append((src, des, {"weight": wgt}))
        return lines


kwargs = {
    'fasta_file': "../data/Fasta/id.fasta"
}
diamond = Diamond("../data/{}".format("Diamond"), **kwargs)
diamond.create_diamond_graph()
