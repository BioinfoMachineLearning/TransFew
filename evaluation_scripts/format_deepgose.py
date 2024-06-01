import os
import pickle
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import networkx as nx


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def read_to_dic(filename):
    res = {}
    with open(filename) as f:
            lines = [line.strip().split("\t") for line in f]

            
    for i in lines:
        prot = i[0].split("|")[1]
        if prot in res:
            res[prot].append((i[1], float(i[2])))
        else:
            res[prot] = [(i[1], float(i[2])), ] 
    return res


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def to_file(data):
    with open('somefile.txt', 'a') as f:
        for ont, prot in data.items():
            f.write("{}\n".format(ont))
            f.write("\t".join(prot))
            f.write("\n")


ROOT_DIR = "/home/fbqc9/Workspace/DATA/"

ontologies = ["cc", "mf", "bp"]
sptr = ['swissprot', 'trembl']
proteins = pickle_load("/home/fbqc9/Workspace/DATA/test/output_t1_t2/test_proteins")

for ont in ontologies:
    print(ont)
    filename = ROOT_DIR + "evaluation/raw_predictions/deepgose/idmapping_2024_04_28_preds_{}.tsv".format(ont)
    data = read_to_dic(filename)

    for st in sptr:
        print(st)
        dir_pth = ROOT_DIR +"evaluation/predictions/{}_{}/".format(st, ont)
        create_directory(dir_pth)

        filt_proteins = proteins[ont][st]

        file_out = open(dir_pth+"{}.tsv".format("deepgose"), 'w')
        for prot in filt_proteins:
            try:
                annots = data[prot]
                for annot in annots:
                    file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
            except KeyError:
                pass
        file_out.close()











exit()









test_group = pickle_load("/home/fbqc9/Workspace/DATA/test/t3/test_proteins")

# add limited known and no known
test_group = {
    'bp': test_group['LK_bp'] | test_group['NK_bp'],
    'mf': test_group['LK_mf'] | test_group['NK_mf'],
    'cc': test_group['LK_cc'] | test_group['NK_cc']
}

to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}



output = {}
for ont in ontologies:
    res = csv_to_dic("evaluation/predictions/deepgose/test_fasta_preds_{}.tsv".format(ont))

    proteins =  set(test_group[ont]).difference(to_remove)

    output[ont] = {key: res[key] for key in proteins}

# to_file(output)

# pickle_save(output, "evaluation/results/{}_out".format("deepgose"))





