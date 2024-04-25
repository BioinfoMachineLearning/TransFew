import os
import pickle
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def csv_to_dic(filename):
    res = {}
    with open(filename) as f:
            lines = [line.split("\t") for line in f]
            
    for i in lines:
        if i[0] in res:
            res[i[0]].append((i[1], float(i[2])))
        else:
            res[i[0]] = [(i[1], float(i[2])), ] 
    return res


ontologies = ["cc", "mf", "bp"]



output = {}
for ont in ontologies:
    res = csv_to_dic("evaluation/predictions/deepgose/test_fasta_preds_{}.tsv".format(ont))
    output[ont] = res

pickle_save(output, "evaluation/results/{}_out".format("deepgose"))