import os
import string
import subprocess
from typing import Tuple, List
import esm
import numpy as np
import torch
from Bio import SeqIO
from scipy.spatial.distance import cdist

import CONSTANTS
from Utils import create_seqrecord

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


# Subsampling MSA
# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)
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


def generate_msa_embedding(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    PDB_IDS = os.listdir(in_dir)
    PDB_IDS = set([i.split(".")[0] for i in PDB_IDS if i.endswith(".a3m")])
    print(len(PDB_IDS))

    generated = os.listdir(out_dir)
    generated = set([i.split(".")[0] for i in generated if i.endswith(".pt")])

    print(len(generated))

    PDB_IDS = PDB_IDS.difference(generated)
    # PDB_IDS = sorted(list(PDB_IDS))
    print(len(PDB_IDS))

    device = 'cuda'
    num_seqs = 128
    seq_len = 1024
    seq_len_min_1 = seq_len - 1

    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval()
    # print(msa_transformer_alphabet.prepend_bos, msa_transformer.append_eos)

    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    for pos, name in enumerate(PDB_IDS):

        try:
            # print("# {}/{}: generating {}".format(pos, len(PDB_IDS), name))
            inputs = read_msa(f"{in_dir}/{name}.a3m")

            ref = inputs[0][1]
            # if len(ref) > 300:
            #     continue

            print("# {}/{}: generating {}".format(pos, len(PDB_IDS), name))

            if len(ref) <= seq_len_min_1:
                inputs = greedy_select(inputs, num_seqs=num_seqs)
                msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = \
                    msa_transformer_batch_converter([inputs])
                msa_transformer = msa_transformer.to(device)
                msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(
                    next(msa_transformer.parameters()).device)
                out = msa_transformer(msa_transformer_batch_tokens, repr_layers=[11, 12], return_contacts=False)

                results = {
                    'label': name,
                    'representations': {
                        11: out['representations'][11][:, :, 1: seq_len, :],
                        12: out['representations'][12][:, :, 1: seq_len, :]
                    },
                    'bos': {
                        11: out['representations'][11][:, :, 0, :],
                        12: out['representations'][12][:, :, 0, :]
                    },
                    'logits': out['logits']
                }
                torch.save(results, f"{out_dir}/{name}.pt")

            else:
                _inputs = greedy_select(inputs, num_seqs=num_seqs)

                cuts = range(int(len(ref) / seq_len_min_1) + 1)

                rep_11 = []
                rep_12 = []
                bos_11 = []
                bos_12 = []
                log_ts = []

                for cut in cuts:
                    inputs = [
                        ('{}_{}'.format(x[0], cut), x[1][cut * seq_len_min_1: (cut * seq_len_min_1) + seq_len_min_1])
                        for x in _inputs]
                    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = \
                        msa_transformer_batch_converter([inputs])
                    msa_transformer = msa_transformer.to(device)
                    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(
                        next(msa_transformer.parameters()).device)
                    out = msa_transformer(msa_transformer_batch_tokens, repr_layers=[11, 12], return_contacts=False)

                    rep_11.append(out['representations'][11][:, :, 1: seq_len, :])
                    rep_12.append(out['representations'][12][:, :, 1: seq_len, :])

                    bos_11.append(out['representations'][11][:, :, 0, :])
                    bos_12.append(out['representations'][12][:, :, 0, :])

                    log_ts.append(out['logits'])

                results = {
                    'label': name,
                    'representations': {
                        11: torch.cat(rep_11, dim=2),
                        12: torch.cat(rep_12, dim=2)
                    },
                    'bos': {
                        11: torch.cat(bos_11, dim=2),
                        12: torch.cat(bos_12, dim=2)
                    },
                    'logits': torch.cat(log_ts, dim=2)
                }

                assert results['representations'][11].shape[2] == results['representations'][12].shape[2] == len(ref)
                torch.save(results, f"{out_dir}/{name}.pt")
        except torch.cuda.OutOfMemoryError:
            pass


def generate_msa_4rm_esm(in_dir, out_dir):
    a3ms = [i.split(".")[0] for i in os.listdir(in_dir)]
    num_seqs = 128
    for a3m in a3ms:
        name = a3m.split(".")[0]

        inputs = read_msa(f"{in_dir}/{name}.a3m")
        all_msas = len(inputs)
        inputs = greedy_select(inputs, num_seqs=num_seqs)

        print("Selected {} out of {}".format(len(inputs), all_msas))

        seqs = []
        for pos, msa in enumerate(inputs):
            seqs.append(create_seqrecord(id="{}_{}".format(name, pos), seq=str(msa[1])))

        fasta_dir = "a3m_fasta/{}/".format(name)
        fasta_pth = fasta_dir + "{}.fasta".format(name)

        os.mkdir("a3m_fasta/{}".format(name))
        SeqIO.write(seqs, fasta_pth, "fasta")

        CMD = "python {} {} {} {} --repr_layers 47 48 --include mean per_tok --nogpu " \
              "--toks_per_batch 1 ".format(CONSTANTS.ROOT + "external/extract.py", "esm2_t48_15B_UR50D",
                                            fasta_pth, fasta_dir)

        subprocess.call(CMD, shell=True, cwd="{}".format(CONSTANTS.ROOT_DIR))

        # stack embeddings
        embeddings = os.listdir(fasta_dir).sort()
        embeddings = [i for i in embeddings if i.endswit(".pt")]


        mean_47 = []
        mean_48 = []
        residue_48 = []
        residue_47 = []
        for emb in embeddings:
            x = torch.load(emb)
            mean_47.append(x['representation']['47'])





generate_msa_4rm_esm("a3ms", "out_dir")
# generate_msa_embedding("a3ms", "out_dir")
# generate_msa_embedding("/home/fbqc9/a3ms", "/home/fbqc9/esm_msa1b")
