import os
import string
from typing import List, Tuple

import torch
from Bio import SeqIO
import numpy as np
from karateclub.dataset import GraphSetReader
from karateclub import FeatherGraph, Graph2Vec, WaveletCharacteristic, LDP
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# reader = GraphSetReader("reddit10k")
#
# graphs = reader.get_graphs()
# y = reader.get_target().tolist()
#
# model = LDP()
# model.fit(graphs)
# X = model.get_embedding()
#
#
# print(X.shape)
#
# exit()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
# y_hat = downstream_model.predict(X_test)[:, :]
#
#
# auc = roc_auc_score(y_test, y_hat)
# print('AUC: {:.4f}'.format(auc))

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


#####
# Check a3ms

# a3ms = os.listdir("../TransFun2Data/data/a3ms")
#
# for a3m in a3ms:
#     _inputs = read_msa(f"{'../TransFun2Data/data/a3ms'}/{a3m}")
#
#     ref = _inputs[0][1]
#     for i in _inputs[1:]:
#         if len(i[1]) > 1021:
#             print(a3m, len(i[1]))
#         assert len(ref) == len(i[1]), "{}_{}_{}".format(a3m, len(ref), len(i[1]))
#     print("######################################333")

# Find ungenerated
# # 0: generating Q06412
# 1307
# # 1: generating 1a3a
# 145
# # 2: generating J9VI03
# 2377

# x = [('1a3a', 1), ('J9VI03', 3), ('Q06412',2)]
#
# for i, j in x:
#     rep_11 = []
#     rep_12 = []
#     logits = []
#
#     for _j in range(j):
#         tmp = torch.load("out_dir/{}_{}.pt".format(i, _j))
#         logits.append(tmp['logits'])
#         rep_11.append(tmp['representations'][11])
#         rep_12.append(tmp['representations'][12])
#         print(i, _j, "logits", tmp['logits'].shape)
#         print(i, _j, 'representations_11', tmp['representations'][11].shape)
#         print(i, _j, 'representations_12', tmp['representations'][12].shape)
#
#     rep11 = torch.cat(rep_11, 2)
#     rep12 = torch.cat(rep_12, 2)
#     logits = torch.cat(logits, 2)
#     print(rep11.shape)
#     print(rep12.shape)
#     print(logits.shape)
#

# x = "AQLQLKISITPRTAWRSRVFVERTALTTRVLTTPPVRVFVERLATALAPSQPRRLRVFVERLATALQEANVSAVLRWDAPEQGQEAPMQALEYHISCWVGSELHEELRLNQSALEARVEHLQPDQTYHFQVEARVAATGAAAGAASHALHVAPEVQAVPRVLYANAEFIGELDLDTRNRRRLVHTASPVEHLVGIEGEQRLLWVNEHVELLTHVPGSAPAKLARMRAEVLALAVDWIQRIVYWAELDATAPQAAIIYRLDLCNFEGKILQGERVWSTPRGRLLKDLVALPQAQSLIWLEYEQGSPRNGSLRGRNLTDGSELEWATVQPLIRLHAGSLEPGSETLNLVDNQGKLCVYDVARQLCTASALRAQLNLLGEAQLQLKISITPRTAWRSGDTTRVQLTTPPVAPSQPRRLRVFVERLATALQEANVSAVLRWDAPEQGQEAPMQALEYHISCWVGSELHEELRLNQSALEARVEHLQPDQTYHFQVEARVAATGAAAGAASHALHVAPEVQAVPRVLYANAEFIGELDLDTRNRRRLVHTASPVEHLVGIEGEQRLLWVNEHVELLTHVPGSAPAKLARMRAEVLALAVDWIQRIVYWAELDATAPQAAIIYRLDLCNFEGKILQGERVWSTPRGRLLKDLVALPQAQSLIWLEYEQGSPRNGSLRGRNLTDGSELEWATVQPLIRLHAGSLEPGSETLNLVDNQGKLCVYDVARQLCTASALRAQLNLLGEAQLQLKISITPRTAWRSGDTTRVQLTTPPVAPSQPRRLRVFVERLATALQEANVSAVLRWEQRLLWVNEHVELLTHVPGSAPAKLARMRAEVLALAVDWIQRIVYWAELDATAPQAAIIYRLDLCNFEGKILQGERVWSTPRGRLLKDLVALPQAQSLIWLEYEQGSPRNGSLRGRNLTDGSELEWATVQPLIRLHAGSLEPGSETLNLVDNQGKLCVYDVARQLCTASALRAQLNLLGEAQLQLKISITPRTAWRSGDTTRVQLTTPPVAPSQPRRLRVFVERLATAL"
# print(len(x))
# exit()

#
#
#     print("###############################33333")

# 'label', 'representations', 'mean_representations', 'contacts'
# 35, 36
x = ['1a3a.pt', 'J9VI03.pt']

for i in x:
    l = torch.load("out_dir/{}".format(i))
    print(l['representations'][11].shape)
    print(l['representations'][12].shape)
    # print(ll['representations'][35].shape)


