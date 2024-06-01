import os
import pickle
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def format_input(fasta_file):
    seqs = []
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for pos, record in enumerate(input_seq_iterator):

        if len(record.seq) > 1000:
            seqs.append(SeqRecord(id=record.id, seq=record.seq[:1000], description=record.description))
        else:
            seqs.append(record)

    SeqIO.write(seqs, "evaluation/predictions/tale/test_fasta.fasta", "fasta")


# format_input("/home/fbqc9/Workspace/DATA/uniprot/test_fasta.fasta")
# exit()
    

def read_files(ont):
    ROOT_DIR = "/home/fbqc9/Workspace/DATA/"
    res = {}
    with open(ROOT_DIR + "evaluation/raw_predictions/tale/{}.out".format(ont)) as f:
            lines = [line.rstrip('\n').replace(")", "")\
                     .replace("(", "")
                     .split("\t") for line in f]
            
    for i in lines:
        term = i[1].replace("'", "").replace(" ", "").split(",")[0]
        if i[0] in res:
            res[i[0]].append((term, float(i[2])))
        else:
            res[i[0]] = [(term, float(i[2])), ]

    return res
            

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        

def write_to_files():

    ROOT_DIR = "/home/fbqc9/Workspace/DATA/"

    ontologies = ["cc", "mf", "bp"]
    sptr = ['swissprot', 'trembl']
    proteins = pickle_load("/home/fbqc9/Workspace/DATA/test/output_t1_t2/test_proteins")

    for ont in ontologies:
        print("Ontology is {}".format(ont))

        data = read_files(ont)


        for st in sptr:
            print("Catehory is {}".format(st))

            dir_pth = ROOT_DIR +"evaluation/predictions/{}_{}/".format(st, ont)
            create_directory(dir_pth)

            filt_proteins = proteins[ont][st]

            file_out = open(dir_pth+"{}.tsv".format("tale"), 'w')
            for prot in filt_proteins:
                annots = data[prot]
                for annot in annots:
                    file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
            file_out.close()


write_to_files()