from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


def extract_id(header):
    return header.split('|')[1]


def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def remove_ungenerated_esm2_daisy_script(fasta_file, generated_directory):
    gen = os.listdir(generated_directory)
    gen = set([i.split(".")[0] for i in gen])

    seq_records = []

    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for record in input_seq_iterator:
        uniprot_id = record.id
        if uniprot_id in gen:
            pass
        else:
            seq_records.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))

    SeqIO.write(seq_records, fasta_file, "fasta")
    # print(len(seq_records), len(gen))

# remove_ungenerated_esm2_daisy_script("/home/fbqc9/EMB/esm2_rem_new.fasta", "/home/fbqc9/esm2_t48")

print(count_proteins("/home/fbqc9/EMB/esm2_rem_new.fasta"))