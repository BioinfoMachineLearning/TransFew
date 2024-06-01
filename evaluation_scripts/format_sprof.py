import os
import pickle
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

ROOT_DIR = "/home/fbqc9/Workspace/DATA/"

def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num

def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        

def combine_predictions():
    path = ROOT_DIR + "evaluation/raw_predictions/sprof/test_{}_all_preds.txt"
    in_files = [900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9900, 10047]

    results = {"cc": {}, "bp": {}, "mf": {} }

    for in_file in in_files:
        print(path.format(in_file))
        with open(path.format(in_file), 'r') as f:
            lines = f.readlines()

        mf_terms, bp_terms, cc_terms = lines[2].strip().split("; "), lines[5].strip().split("; "), lines[8].strip().split("; ")

        assert lines[10] == "\n"

        for line in lines[10:]:
            if line == "\n":
                pass
            elif line == "MF:\n":
                cur = "_mf"
            elif line == "CC:\n":
                cur = "_cc"
            elif line == "BP:\n":
                cur = "_bp"
            else:
                split_line = line.strip().split(";")
                if len(split_line) == 1:
                    protein = split_line[0]
                elif cur == "_mf":
                    assert len(split_line) == len(mf_terms)
                    split_line = [float(i) for i in split_line]
                    results['mf'][protein] = list(zip(mf_terms, split_line))
                elif cur == "_cc":
                    assert len(split_line) == len(cc_terms)
                    split_line = [float(i) for i in split_line]
                    results['cc'][protein] = list(zip(cc_terms, split_line))
                elif cur == "_bp":
                    assert len(split_line) == len(bp_terms)
                    split_line = [float(i) for i in split_line]
                    results['bp'][protein] = list(zip(bp_terms, split_line))

    return results







ontologies = ["cc", "mf", "bp"]
sptr = ['swissprot', 'trembl']
proteins = pickle_load(ROOT_DIR + "test/output_t1_t2/test_proteins")

all_data = combine_predictions()


for ont in ontologies:
    print("Ontology is {}".format(ont))

    data = all_data[ont]

    for st in sptr:
        print("Category is {}".format(st))

        dir_pth = ROOT_DIR +"evaluation/predictions/full/{}_{}/".format(st, ont)
        create_directory(dir_pth)

        filt_proteins = proteins[ont][st]


        file_out = open(dir_pth+"{}.tsv".format("sprof"), 'w')
        for prot in filt_proteins:
            annots = data[prot]
            for annot in annots:
                file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
        file_out.close()





