import os
import shutil
import subprocess
from matplotlib import pyplot as plt, rcParams
import numpy as np
from Utils import count_proteins, create_directory, pickle_save, pickle_load
from collections import Counter
import math
import CONSTANTS
import obonet
import networkx as nx
from Bio import SeqIO


def filter_fasta(proteins, infile, outfile):
    seqs = []
    input_seq_iterator = SeqIO.parse(infile, "fasta")

    for pos, record in enumerate(input_seq_iterator):
        if record.id in proteins:
            seqs.append(record)
    SeqIO.write(seqs, outfile, "fasta")


def extract_from_results(infile):
    file = open(infile)
    lines = []
    for _line in file.readlines():
        line = _line.strip("\n").split("\t")
        lines.append((line[0], line[1], line[3]))
    file.close()
    return lines


def get_seq_less(ontology, test_proteins, seq_id=0.3):
    # mmseqs createdb <i:fastaFile1[.gz|.bz2]> ... <i:fastaFileN[.gz|.bz2]>|<i:stdin> <o:sequenceDB> [options]

    full_train_fasta = "/home/fbqc9/Workspace/DATA/uniprot/train_sequences.fasta"
    test_fasta = "/home/fbqc9/Workspace/DATA/uniprot/test_fasta.fasta"

    # Number of training proteins & test proteins in combined fasta
    # print("# Training: {}, # Testing {}".format(count_proteins(full_train_fasta), count_proteins(test_fasta)))

    train_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/train_proteins".format(ontology)))
    valid_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/validation_proteins".format(ontology)))


    train_data = set(train_data + valid_data) # set for fast lookup
    test_proteins = set(test_proteins) # set for fast lookup

    # Number of training proteins & test proteins
    print("# Training & Validation: {}, # Testing {}".format(len(train_data), len(test_proteins)))
    
    # No train data in test data
    assert len(train_data.intersection(test_proteins)) == 0 


    # make temporary directory
    wkdir = "/home/fbqc9/Workspace/TransFun2/evaluation/seqID/{}".format(seq_id)
    create_directory(wkdir)

    # create query and target databases
    target_fasta = wkdir+"/target_fasta"
    query_fasta = wkdir+"/query_fasta"
    filter_fasta(train_data, full_train_fasta, target_fasta)
    filter_fasta(test_proteins, test_fasta, query_fasta)

    # All train and test in respective fasta
    assert len(train_data) == count_proteins(target_fasta)
    assert len(test_proteins) == count_proteins(query_fasta)

    print("Creating target Database")
    target_dbase = wkdir+"/target_dbase"
    CMD = "mmseqs createdb {} {}".format(target_fasta, target_dbase)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Creating query Database")
    query_dbase = wkdir+"/query_dbase"
    CMD = "mmseqs createdb {} {}".format(query_fasta, query_dbase)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Mapping very similar sequences")
    result_dbase = wkdir+"/result_dbase"
    CMD = "mmseqs map {} {} {} {} --min-seq-id {}".\
        format(query_dbase, target_dbase, result_dbase, wkdir, seq_id)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    bestResultDB = wkdir+"/bestResultDB"
    CMD = "mmseqs filterdb {} {} --extract-lines 1".format(result_dbase, bestResultDB)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    final_res = wkdir+"/final_res.tsv"
    CMD = "mmseqs createtsv {} {} {} {}".format(query_dbase, target_dbase, bestResultDB, final_res)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    lines = extract_from_results(final_res)

    shutil.rmtree(wkdir)

    querys, targets, seq_ids = zip(*lines)

    querys = set(querys)
    targets = set(targets)

    assert len(train_data.intersection(querys)) == 0
    assert len(test_proteins.intersection(targets)) == 0

    assert len(train_data.intersection(targets)) == len(targets)
    assert len(test_proteins.intersection(querys)) == len(querys)


    # the proteins with less than X seq identity to the training set
    return test_proteins.difference(querys)


def read_filter_write(proteins, in_file, out_file):

    with open(in_file) as f:
        lines = [line.strip('\n').split("\t") for line in f]

    lines = [i for i in lines if i[0] in proteins]

    '''file_out = open(out_file, 'w')
    file_out.write('\t'.join(lines) + '\n')
    file_out.close()'''

    file_out = open(out_file, 'w')
    for line in lines:
        file_out.write('\t'.join(line) + '\n')
        # file_out.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
    file_out.close()


all_proteins = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/test_proteins")

methods = ['naive', 'diamond', 'tale', 'deepgose', 'netgo3', 'sprof', 'transfew']

in_file_pths = CONSTANTS.ROOT_DIR + "evaluation/predictions/full/{}_{}/{}.tsv"
out_file_pths = CONSTANTS.ROOT_DIR + "evaluation/predictions/seq_ID_30/{}_{}/"

gt_in_file_pths = CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruths/full/{}_{}.tsv"
gt_out_file_pths = CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruths/seq_ID_30/{}_{}.tsv"

def main():
    for ont in all_proteins:
        for sptr in all_proteins[ont]:
        
            proteins = all_proteins[ont][sptr]
            proteins = get_seq_less(ontology=ont, test_proteins=proteins)

            print("Writing groundtruth {} {}".format(ont, sptr))
            read_filter_write(proteins, gt_in_file_pths.format(ont, sptr), gt_out_file_pths.format(ont, sptr))

            create_directory(out_file_pths.format(sptr, ont))
            # write output from all_output
            for method in methods:

                print("Ontology: {} --- DB subset {} --- Method {}".format(ont, sptr, method))
                read_filter_write(proteins, in_file_pths.format(sptr, ont, method), out_file_pths.format(sptr, ont) + "{}.tsv".format(method))
                


if __name__ == '__main__':

    main()

    