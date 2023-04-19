import os
import pickle

import torch
from Bio import SeqIO


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Store ESM embeddings into dictionary
def embedding_to_dic(outname, dir):
    emb_files = os.listdir(dir)

    results = {}
    for pos, emb in enumerate(emb_files):
        print(pos, emb)
        x = torch.load(dir + "/{}".format(emb))
        results[x['label']] = {
            'representations_48': x['representations'][48],
            'mean_representations_48': x['mean_representations'][48]
        }

    pickle_save(results, outname)

embedding_to_dic("/home/fbqc9/esm2_t48", "res")

# embedding_to_dic("D:\Workspace\python-3\TransFun2\processed\esm2_t48_rem",
#                  "D:\Workspace\python-3\TransFun2\processed\esm2_t48_rem")

# def combine_split_embeddings(files):
#     for i in lines:
#         pr

# x = pickle_load("../../prots_gt_1022")
# print(x)


exit()


def fasta_to_dictionary(fasta_file, identifier='protein_id'):
    if identifier == 'protein_id':
        loc = 1
    elif identifier == 'protein_name':
        loc = 2
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        if "|" in seq_record.id:
            data[seq_record.id.split("|")[loc]] = (
                seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
        else:
            data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def get_proteins_from_fasta(fasta_file):
    proteins = list(SeqIO.parse(fasta_file, "fasta"))
    proteins = [i.id for i in proteins]
    return proteins


full_fasta = fasta_to_dictionary("/home/fbqc9/EMB/uniprot_fasta.fasta")
x = get_proteins_from_fasta("/home/fbqc9/EMB/uniprot_gt_1022.fasta")
keys = {}

for protein in x:
    _temp = protein.split("_")[0]
    if _temp in keys:
        keys[_temp] = keys[_temp] + 1
    else:
        keys[_temp] = 1

embeddings = [47, 48]
for protein in keys:
    tmp = []
    results = {'representations': {}, 'mean_representations': {}, 'label': ''}
    for cut in range(keys[protein]):
        x = torch.load("/home/fbqc9/esm2_t48_rem/{}_{}.pt".format(protein, cut))

        if results['label'] == '':
            results['label'] = x['label'].split("_")[0]
        else:
            assert results['label'] == x['label'].split("_")[0]

        tmp.append(x)

    for index in tmp:
        for rep in embeddings:
            assert torch.equal(index['mean_representations'][rep], torch.mean(index['representations'][rep], dim=0))

            if rep in results['representations']:
                results['representations'][rep] = torch.cat(
                    (results['representations'][rep], index['representations'][rep]))
            else:
                results['representations'][rep] = index['representations'][rep]

    for emb in embeddings:
        assert len(full_fasta[protein][3]) == results['representations'][emb].shape[0]

    for rep in embeddings:
        results['mean_representations'][rep] = torch.mean(results['representations'][rep], dim=0)

    torch.save(results, "/home/fbqc9/emb_join/{}.pt".format(protein))
