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
    

# divide data into chunks of 1000
def divide(fasta_file):
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    seqs = []
    for pos, record in enumerate(input_seq_iterator):
        seqs.append(record)
        if pos % 900 == 0 and pos>0:
            SeqIO.write(seqs, "/home/fbqc9/Workspace/TransFun2/evaluation/predictions/netgo/raw/{}".format('test_{}.fasta'.format(pos)), "fasta")
            seqs = []
    SeqIO.write(seqs, "/home/fbqc9/Workspace/TransFun2/evaluation/predictions/netgo/raw/{}".format('test_{}.fasta'.format(pos)), "fasta")

# divide("/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta")


def combine_predictions():
    path = "evaluation/predictions/netgo/result_{}.txt"
    results = [1699845719, 1699845761, 1699845770, 1699845780, 1699897400,
               1705853012, 1705853252, 1705853349, 1705908596, 1705908641,
               1705908662, 1705908663]

    with open("evaluation/predictions/netgo/combined_result.txt", 'w') as outfile:
        for fname in results:
            with open(path.format(fname)) as infile:
                outfile.write(infile.read())



def read_files():

    res_dic = {'mf':{}, 'cc':{}, 'bp':{}}
    ROOT_DIR = "/home/fbqc9/Workspace/DATA/"
    all_netgo_output = [1699845719, 1699845761, 1699845770, 1699845780, 1699897400,
               1705853012, 1705853252, 1705853349, 1705908596, 1705908641,
               1705908662, 1705908663]
    # test_group = pickle_load("/home/fbqc9/Workspace/DATA/test/t3/test_proteins")


    for netgo_output in all_netgo_output:
        with open(ROOT_DIR + "evaluation/raw_predictions/netgo/{}.txt".format("result_{}".format(netgo_output))) as f:
            lines = [line.rstrip('\n').split("\t") for line in f]
            
            for i in lines:
                if i[0] == "=====":
                    continue
                if i[0] in res_dic[i[3]]:
                    res_dic[i[3]][i[0]].append((i[1], float(i[2])))
                else:
                    res_dic[i[3]][i[0]] = [(i[1], float(i[2])), ]

    return res_dic


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def write_to_files():

    all_data = read_files()

    ROOT_DIR = "/home/fbqc9/Workspace/DATA/"

    ontologies = ["cc", "mf", "bp"]
    sptr = ['swissprot', 'trembl']
    proteins = pickle_load(ROOT_DIR + "test/output_t1_t2/test_proteins")

    for ont in ontologies:
        print("Ontology is {}".format(ont))
        data = all_data[ont]

        for st in sptr:
            print("Catehory is {}".format(st))

            dir_pth = ROOT_DIR +"evaluation/predictions/{}_{}/".format(st, ont)
            create_directory(dir_pth)

            filt_proteins = proteins[ont][st]

            file_out = open(dir_pth+"{}.tsv".format("netgo3"), 'w')
            for prot in filt_proteins:
                annots = data[prot]
                for annot in annots:
                    file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
            file_out.close()



write_to_files()