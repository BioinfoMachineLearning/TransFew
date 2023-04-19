import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def is_file(path):
    return os.path.exists(path)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def extract_id(header):
    return header.split('|')[1]


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


def remove_ungenerated_esm2_daisy_script(fasta_file, generated_directory):
    import os
    # those generated
    gen = os.listdir(generated_directory)
    gen = set([i.split(".")[0] for i in gen])

    seq_records = []

    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for record in input_seq_iterator:
        uniprot_id = extract_id(record.id)
        seq_records.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))

    print(len(seq_records), len(gen), len(set(seq_records).difference(gen)))

def filtered_sequences(fasta_file):
    """
         Script is used to create fasta files based on alphafold sequence, by replacing sequences that are different.
        :param uniprot_fasta_file: input uniprot fasta file.
        :return: None
        """

    seq_records = []

    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for record in input_seq_iterator:
        uniprot_id = extract_id(record.id)
        seq_records.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))

    SeqIO.write(seq_records, "data/Fasta/id2.fasta", "fasta")


def readlines_cluster(in_file):
    file = open(in_file)
    lines = [line.strip("\n").split("\t") for line in file.readlines() if line.strip()]
    file.close()
    return lines

# print(count_proteins("data/Fasta/filtered.fasta"))

# filtered_sequences("data/Fasta/cleaned.fasta")