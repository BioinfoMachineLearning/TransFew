import os
import pickle
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def read_tsv(file_name):
    results = {}
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]

    for line in lines:
        if line[0] in results:
            results[line[0]].append((line[1], line[2]))
        else:
            results[line[0]] = [(line[1], line[2]), ]
    return results


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)
    

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


ontologies = ["cc", "bp", "mf"]
db_subset = ['swissprot', 'trembl']
methods = ["esm2_t48", "msa_1b", "interpro", "split_mlp1", "split_mlp2", "combined", "transfew"]
# "slpit_mlp3"


def write_to_file(methods, out_dir):

    ROOT_DIR = "/home/fbqc9/Workspace/DATA/"
    proteins = pickle_load(ROOT_DIR + "test/output_t1_t2/test_proteins")
    RAW_PRED_pth = ROOT_DIR + "evaluation/raw_predictions/{}/{}.tsv"
    

    for method in methods:
        for ont in ontologies[1:2]:
            results = read_tsv(RAW_PRED_pth.format(method, ont))
            for sptr in db_subset:
                tmp_proteins = proteins[ont][sptr]
            
                print(ont, sptr, len(results), len(tmp_proteins))

                dir_pth = ROOT_DIR +"evaluation/predictions/{}/{}_{}/".format(out_dir, sptr, ont)
                create_directory(dir_pth)

                file_out = open(dir_pth+"{}.tsv".format(method), 'w')
                for prot in results:
                    if prot in tmp_proteins:
                        annots = results[prot]
                        for annot in annots:
                            file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
                file_out.close()


# Write full predictions to file to compare with other methods.
methods = ["transfew"]
# write_to_file(methods=methods, out_dir="full")


# write the combined:
methods = ["esm2_t48", "msa_1b", "interpro", "combined", "transfew"]
# write_to_file(methods=methods, out_dir="components")


# write random split:
methods = ["split_mlp1", "split_mlp2", "split_mlp3", "transfew"]
# write_to_file(methods=methods, out_dir="random_split")