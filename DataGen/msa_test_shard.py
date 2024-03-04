import os
import string
from typing import Tuple, List
import esm
import numpy as np
import torch
from Bio import SeqIO
import traceback
from scipy.spatial.distance import cdist

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

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap, auto_wrap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

def generate_msa_embedding(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    a3ms = os.listdir(in_dir)
    a3ms = set([i.split(".")[0] for i in a3ms if i.endswith(".a3m")])
    
    generated = os.listdir("/bmlfast/frimpong/shared_function_data/esm_msa1b")
    generated = set([i.split(".")[0] for i in generated])
    print(len(a3ms), len(generated), len(a3ms.difference(generated)))


    
    #generated = generated.union(generated2)

    a3ms = a3ms.difference(generated)
    a3ms = sorted(list(a3ms))#[0:2500]
    # PDB_IDS.reverse()
    print(len(a3ms))


    device = 'cuda:0'
    num_seqs = 128
    seq_len = 1024
    seq_len_min_1 = seq_len - 1

    # init the distributed world with world_size 1
    url = "tcp://localhost:23455"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    # initialize the model with FSDP wrapper
    fsdp_params = dict(wrapper_cls=FSDP,
                       compute_device=device,
                        mixed_precision=True,
                        flatten_parameters=True,
                        #state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                        cpu_offload=False
                        )  # enable cpu offloading

    with enable_wrap(**fsdp_params):
        msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
        msa_transformer.eval()

        # msa_transformer = msa_transformer.eval()#.to(device)
        

        for name, child in msa_transformer.named_children():
            
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        msa_transformer = wrap(msa_transformer)

    for pos, name in enumerate(a3ms):

        try:
            inputs = read_msa(f"{in_dir}/{name}.a3m")
            ref = inputs[0][1]
            print("# {}/{}: generating {}, lenth {}".format(pos, len(a3ms), name, len(ref)))

            if len(ref) <= seq_len_min_1:
                inputs = greedy_select(inputs, num_seqs=num_seqs)
                msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = \
                    msa_transformer_batch_converter([inputs])
                # msa_transformer = msa_transformer.to(device)
                msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).to(device)).long()
                #msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(device)
                out = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=False)

                results = {
                    'label': name,
                    'representations_12': out['representations'][12][:, :, 1: seq_len, :]
                    }
                assert results['representations_12'].shape[2] == len(ref)
                results['representations_12'] = results['representations_12'].mean(2)
                torch.save(results, f"{out_dir}/{name}.pt")
                del out
                del results
                print("generated")

            else:
                _inputs = greedy_select(inputs, num_seqs=num_seqs)

                cuts = range(int(len(ref)/seq_len_min_1) + 1)

                rep_12 = []
                log_ts = []

                for cut in cuts:
                    inputs = [('{}_{}'.format(x[0], cut), x[1][cut*seq_len_min_1: (cut*seq_len_min_1) + seq_len_min_1]) for x in _inputs]
                    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = \
                        msa_transformer_batch_converter([inputs])
                    #msa_transformer = msa_transformer.to(device)
                    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device).long()
                    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(device)
                    out = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=False)

                    torch.save(out, f"{out_dir}/{name}_{cut}.pt")
                    
                    del out

                for cut in cuts:
                    out = torch.load(f"{out_dir}/{name}_{cut}.pt")
                    rep_12.append(out['representations'][12][:, :, 1: seq_len, :])
                    log_ts.append(out['logits'])

                del out

                results = {
                    'label': name,
                    'representations_12': torch.cat(rep_12, dim=2)
                }

                assert results['representations_12'].shape[2] == len(ref)
                results['representations_12'] = results['representations_12'].mean(2)
                torch.save(results, f"{out_dir}/{name}.pt")
                del results

                for cut in cuts:
                    os.remove(f"{out_dir}/{name}_{cut}.pt")
                print("generated")
        except Exception as e:
            print(traceback.format_exc())


# add aggragation


# /bmlfast/frimpong/shared_function_data/
# generate_msa_embedding("/bmlfast/frimpong/shared_function_data/a3ms", "/bmlfast/frimpong/shared_function_data/esm_msa1b")
generate_msa_embedding("/bmlfast/frimpong/shared_function_data/a3ms", "/home/fbqc9/msa_new")
# generate_msa_embedding("/home/fbqc9/a3ms", "/home/fbqc9/esm_msa1b")
