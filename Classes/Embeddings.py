import csv
import subprocess

import Constants
from Utils import is_file, create_directory, readlines_cluster


class Embeddings:
    """
    This class is used to handle all embedding generations
    """

    def __init__(self, **kwargs):
        self.dir = kwargs.get('dir', Constants.ROOT + "data/Embeddings/")
        self.fasta = kwargs.get('fasta', None)
        self.session = "training"
        self.database = kwargs.get('database', "")
        self.run()

    def create_database(self):
        dbase_name = self.dir + "database/mmseq_dbase"
        if not is_file(dbase_name):
            create_directory(self.dir + "database/")
            CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat createdb {} {}"
            subprocess.call(CMD.format(self.fasta, dbase_name), shell=True, cwd="{}".format(self.dir))

    def one_line_format(self, input_file):
        """
             Script takes the mm2seq cluster output and converts to representative seq1, seq2, seq3 ....
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
        with open(self.dir + "cluster/final_clusters.csv", "w") as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerows(result)

    # cluster data
    def generate_cluster(self):
        cluster_name = self.dir + "cluster/mmseq_cluster"
        dbase_name = self.dir + "database/mmseq_dbase"
        final_cluster = cluster_name + ".tsv"
        if not is_file(final_cluster):
            create_directory(self.dir + "cluster/")
            CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat cluster {} {} tmp".format(dbase_name, cluster_name)
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

            CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat createtsv {} {} {} {}.tsv".format(dbase_name,
                                                                                                  dbase_name,
                                                                                                  cluster_name,
                                                                                                  cluster_name)
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

            # CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat cluster {} {} tmp;" \
            #       "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat createtsv {} {} {} {}.tsv" \
            #     .format("dbase_name", "cluster_name", "dbase_name", "dbase_name", "cluster_name", "dbase_name" + ".tsv")

            self.one_line_format(final_cluster)

    # generate msa from cluster
    def msa_from_cluster(self):
        cluster_name = self.dir + "cluster/mmseq_cluster"
        dbase_name = self.dir + "database/mmseq_dbase"
        msa_name = cluster_name + "_msa"
        if not is_file(msa_name):
            CMD = "D:/Workspace/python-3/TFUN/mmseqs/mmseqs.bat result2msa {} {} {} {}" \
                .format(dbase_name, dbase_name, cluster_name, cluster_name + "_msa")
            subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

    def create_msa_files(self):
        lines = readlines_cluster("D:/Workspace/python-3/TFUN/data/Embeddings/cluster/final_clusters.csv")
        print(lines)

    def search(self, sequences):
        pass

    # generate embeddings
    def generate_embeddings(self):
        models = (("esm_msa_1b", "esm_msa1b_t12_100M_UR50S", "msa", "output", 11, 12),
                  ("esm_2", "esm2_t48_15B_UR50D", self.fasta, "output", 47, 48),
                  ("esm_1b", "esm2_t36_3B_UR50D", self.fasta, "output", 35, 36))
        for model in models[1:]:
            if model[0] == "esm_msa_1b":
                CMD = "python D:/Workspace/python-3/TFUN/external/extract.py {} {} {} --repr_layers {} {} --include mean per_tok --toks_per_batch 2 " \
                    .format(model[1], model[2], model[3], model[4], model[5])
                print(CMD)
                subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))
            else:
                CMD = "python D:/Workspace/python-3/TFUN/external/extract.py {} {} {} --repr_layers {} {} --include mean per_tok --toks_per_batch 2 " \
                    .format(model[1], model[2], model[3], model[4], model[5])
                print(CMD)
                subprocess.call(CMD, shell=True, cwd="{}".format(self.dir))

    def run(self):
        if self.session == "training":
            self.create_database()
            # self.generate_cluster()
            # self.msa_from_cluster()
            self.generate_embeddings()
            # self.create_msa_files()
            # self.search()
        else:
            pass
            # search
            # generate embeddings

        # bulk embedding from fasta(per-residue + per-sequence)
        # bulk embedding from msa


kwargs = {
    'fasta': Constants.ROOT + "data/Fasta/id.fasta"
}
embeddings = Embeddings(**kwargs)
