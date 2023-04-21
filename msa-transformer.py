import numpy as np
from biotransformers import BioTransformers
from biotransformers.utils.msa_utils import msa_to_remove
import esm, torch

# msa_folder = "D:/Workspace/python-3/TransFun2/a3ms"

# msatr = msa_to_remove(msa_folder, n_seq=128)
# print(msatr)
#
#
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
# ]
#
# bio_trans = BioTransformers("esm1_t6_43M_UR50S", num_gpus=0)
# msa_embeddings = bio_trans.compute_embeddings(sequences=data, pool_mode=("full"), n_seqs_msa=128)
# print(msa_embeddings['full'][0].shape)
#
# print(msa_embeddings['full'][1])
#
# print(msa_embeddings['cls'].shape)

# Load ESM-2 model
# esm1_t6_43M_UR50S
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1_t6_43M_UR50S")
model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6], return_contacts=True)
token_representations = results["representations"][6]

sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

print(np.array(sequence_representations[0]))