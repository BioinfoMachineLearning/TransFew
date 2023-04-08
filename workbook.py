# Notebook to do surplus work
import os
import shutil

import pandas as pd

import CONSTANTS
from Utils import count_proteins

# def interpro_go_terms():
#     with open(CONSTANTS.ROOT_DIR + "interpro/interpro2go") as file:
#         next(file)
#         next(file)
#         next(file)
#         next(file)
#         next(file)
#         terms = {}
#         for line in file:
#             go = line.strip().split(";")[1]
#             ipr = line.strip().split(",")[0].split(" ")[0].split(":")[1]
#
#             if go in terms:
#                 print(go)
#                 terms[ipr].append(go)
#             else:
#                 terms[ipr] = [go]
#
#     print(terms)
#
#
# interpro_go_terms()


# print(count_proteins("/home/fbqc9/Desktop/hua/Helitron_sequences.fasta"))
#
# print(count_proteins("/home/fbqc9/Desktop/hua/PIF1-like sequences.fasta"))

data = pd.read_csv(CONSTANTS.ROOT_DIR + "training.csv")
print(data.shape)