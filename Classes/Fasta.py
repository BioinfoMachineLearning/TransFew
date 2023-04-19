from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import CONSTANTS


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


def extract_id(header):
    return header.split('|')[1]


class Fasta:
    def __init__(self, fasta_path):
        self.fasta = fasta_path

    def fasta_to_list(self):
        seqs = []
        input_seq_iterator = SeqIO.parse(self.fasta, "fasta")
        for record in input_seq_iterator:
            seqs.append(create_seqrecord(id=record.id, seq=str(record.seq)))
        return seqs
    
    def reformat(self):
        seqs = self.fasta_to_list()
        SeqIO.write(seqs, CONSTANTS.ROOT_DIR + "uniprot/{}.fasta".format("uniprot_fasta"), "fasta")

    # Create individual fasta files from a large fasta file for hhblits alignment
    def fastas_from_fasta(self, _filter=None):
        seqs = self.fasta_to_list()
        for seq in seqs:
            SeqIO.write(seq, CONSTANTS.ROOT_DIR + "uniprot/single_fasta/{}.fasta".format(seq.id), "fasta")

    # Removes unwanted proteins from fasta file
    def subset_from_fasta(self, output=None, max_seq_len=1022):
        seqs = self.fasta_to_list()
        subset = []
        for record in seqs:
            if len(record.seq) > max_seq_len:
                splits = range(int(len(record.seq)/max_seq_len) + 1)

                for split in splits:
                    seq_len = str(record.seq[split*max_seq_len: (split*max_seq_len) + max_seq_len])

                    subset.append(create_seqrecord(id='{}_{}'.format(record.id, split), seq=seq_len))

        SeqIO.write(subset, output, "fasta")

    # Count the number of protein sequences in a fasta file with biopython -- slower.
    def count_proteins_biopython(self):
        num = len(list(SeqIO.parse(self.fasta, "fasta")))
        return num




# fasta_path = CONSTANTS.ROOT_DIR + "uniprot/uniprot_fasta.fasta"
# embeddings = Fasta(fasta_path)
# # embeddings.reformat()
# embeddings.fastas_from_fasta()
