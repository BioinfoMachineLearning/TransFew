import sys
import csv
import subprocess
from os import listdir

import pandas as pd
from Bio import SeqIO
import CONSTANTS
from Utils import is_file, create_directory, readlines_cluster, extract_id
from preprocessing.utils import create_seqrecord


class Embeddings:
    """
    This class is used to handle all embedding generations
    """

    def __init__(self, **kwargs):
        self.dir = kwargs.get('dir', CONSTANTS.ROOT_DIR)
        self.fasta = kwargs.get('fasta', None)
        self.session = "training"
        self.database = kwargs.get('database', "")

        self.mmseq_root = self.dir + "mmseq/"
        self.mmseq_dbase_root = self.mmseq_root + "dbase/"
        self.mmseq_cluster_root = self.mmseq_root + "cluster/"

        self.uniclust_dbase = None

    def create_database(self):
        mmseq_dbase_path = self.mmseq_dbase_root + "mmseq_dbase"
        if not is_file(mmseq_dbase_path):
            create_directory(self.mmseq_dbase_root)
            CMD = "mmseqs createdb {} {}".format(self.fasta, mmseq_dbase_path)
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))
            print("mmseq database created")

    def generate_cluster(self):
        mmseq_dbase_path = self.mmseq_dbase_root + "mmseq_dbase"
        mmseq_cluster_path = self.mmseq_cluster_root + "mmseq_cluster"
        final_cluster = mmseq_cluster_path + ".tsv"
        output = self.mmseq_cluster_root + "final" + ".tsv"
        if not is_file(final_cluster):
            create_directory(self.mmseq_cluster_root)
            CMD = "mmseqs cluster {} {} tmp ; " \
                  "mmseqs createtsv {} {} {} {}.tsv".format(mmseq_dbase_path, mmseq_cluster_path, mmseq_dbase_path,
                                                            mmseq_dbase_path, mmseq_cluster_path, mmseq_cluster_path)

            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

            self.one_line_format(final_cluster, output)

    @staticmethod
    def one_line_format(input_file, output):
        """
            Script takes the mm2seq cluster output and converts to representative seq1, seq2, seq3 ....
            :param output:
            :param input_file: The clusters as csv file
            :return: None
        """
        data = {}
        with open(input_file) as file:
            lines = file.read().splitlines()
            for line in lines:
                x = line.split("\t")
                if x[0] in data:
                    data[x[0]].append(x[1])
                else:
                    data[x[0]] = list([x[1]])
        result = [data[i] for i in data]
        with open(output, "w") as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerows(result)

    def generate_msas(self):
        # this part was run on lotus
        all_fastas = listdir(CONSTANTS.ROOT_DIR + "uniprot/single_fasta/")
        for fasta in all_fastas:
            fasta_name = fasta.split(".")[0]
            CMD = "hhblits -i query.fasta -d {} -oa3m msas/{}.a3m -cpu 4 -n 2".format(self.uniclust_dbase, fasta_name)
            print(CMD)
            # subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

    # generate msa from cluster
    def msa_from_cluster(self):

        cluster_name = self.dir + "cluster/mmseq_cluster"
        dbase_name = self.dir + "database/mmseq_dbase"
        msa_name = cluster_name + "_msa"
        if not is_file(msa_name):
            CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat result2msa {} {} {} {}" \
                .format(dbase_name, dbase_name, cluster_name, cluster_name + "_msa")
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

    def read_fasta_list(self):
        training_proteins = set(pd.read_csv("../preprocessing/uniprot.csv", sep="\t")['ACC'].to_list())
        seqs = []
        input_seq_iterator = SeqIO.parse(self.fasta, "fasta")
        for record in input_seq_iterator:
            uniprot_id = extract_id(record.id)
            if uniprot_id in training_proteins:
                seqs.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))
        return seqs

    def search(self, sequences):
        pass

    # generate embeddings
    def generate_embeddings(self):
        # name model output dir, embedding layer 1, embedding layer 2, batch
        models = (
        ("esm_msa_1b", "esm_msa1b_t12_100M_UR50S", "msa", CONSTANTS.ROOT_DIR + "embedding/esm_msa_1b", 11, 12, 10),
        ("esm_2", "esm2_t48_15B_UR50D", self.fasta, CONSTANTS.ROOT_DIR + "embedding/esm2_t48", 47, 48, 10),
        ("esm_2", "esm2_t36_3B_UR50D", self.fasta, CONSTANTS.ROOT_DIR + "embedding/esm_t36", 35, 36, 100))
        for model in models[2:]:
            if model[0] == "esm_msa_1b":
                CMD = "python {} {} {} {} --repr_layers {} {} --include mean per_tok " \
                      "--toks_per_batch {} ".format(CONSTANTS.ROOT + "external/extract.py", model[1], model[2],
                                                    model[3], model[4], model[5], model[6])
            else:
                CMD = "python {} {} {} {} --repr_layers {} {} --include mean per_tok --nogpu " \
                      "--toks_per_batch {} ".format(CONSTANTS.ROOT + "external/extract.py", model[1], model[2],
                                                    model[3], model[4], model[5], model[6])

            print(CMD)
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

    def run(self):
        self.create_database()
        self.generate_cluster()
