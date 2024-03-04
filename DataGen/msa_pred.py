from Bio import SeqIO
import traceback
from scipy.spatial.distance import cdist
from typing import Tuple, List, Set
import numpy as np
import string
from Bio.UniProt import GOA
import os
import pickle


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def get_id_msa(id: str) -> str:
    try:
        return id.split("_")[1]
    except IndexError:
        return id

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(get_id_msa(record.id), remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


# Subsampling MSA
# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int , valid_prots: Set[str], mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for id, seq in msa if id in valid_proteins], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))

    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


def read_gaf(handle):
    dic = {}
    all_protein_name = set()
    # evidence from experimental
    Evidence = {'Evidence': set(["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"])}
    with open(handle, 'r') as handle:
        for rec in GOA.gafiterator(handle):
            if rec['DB'] == 'UniProtKB':
                all_protein_name.add(rec['DB_Object_ID'])
                if rec['DB_Object_ID'] not in dic:
                    dic[rec['DB_Object_ID']] = {rec['Aspect']: set([rec['GO_ID']])}
                else:
                    if rec['Aspect'] not in dic[rec['DB_Object_ID']]:
                        dic[rec['DB_Object_ID']][rec['Aspect']] = set([rec['GO_ID']])
                    else:
                        dic[rec['DB_Object_ID']][rec['Aspect']].add(rec['GO_ID'])
    return dic, all_protein_name



# cv = pickle_load("proteins_xxx_proteins")



# print(len(cv))

# exit()

# valid_proteins = os.listdir("/home/fbqc9/a3ms")
# valid_proteins = set([i.split(".")[0] for i in valid_proteins])

# print(len(valid_proteins))

# exit()
data, proteins = read_gaf("/home/fbqc9/Workspace/DATA/uniprot/goa_uniprot_all.gaf.212")

pickle_save(data, "data_xxx_data")

pickle_save(proteins, "proteins_xxx_proteins")



exit()
num_seqs = 10

inputs = read_msa("/home/fbqc9/a3ms/{}.a3m".format("Q54801"))


valid_proteins = os.listdir("/home/fbqc9/a3ms")
print(valid_proteins[:10])
valid_proteins = set([i.split(".")[0] for i in valid_proteins])

inputs = greedy_select(inputs, num_seqs=num_seqs, valid_prots=valid_proteins)