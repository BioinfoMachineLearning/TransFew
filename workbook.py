# Notebook to do surplus work
import os
import shutil

from preprocessing.utils import pickle_load


# Just copy remaining proteins for msa
def my_copy():
    generated = os.listdir("/home/fbqc9/PycharmProjects/TransFun2Data/uniprot/a3ms")
    generated = set([i.split(".")[0] for i in generated])

    remaining = set(os.listdir("/home/fbqc9/PycharmProjects/TransFun2Data/uniprot/single_fasta2"))
    remaining = set([i.split(".")[0] for i in remaining])

    remaining = remaining.difference(generated)
    print(len(remaining))


    for i in remaining:
        print(i)
        src = "/home/fbqc9/PycharmProjects/TransFun2Data/uniprot/single_fasta2/{}.fasta".format(i)
        dst = "/home/fbqc9/PycharmProjects/TransFun2Data/uniprot/single_fasta/{}.fasta".format(i)
        shutil.copyfile(src, dst)

# my_copy()


data = pickle_load("test")

for i in data['cc']:
    print(i, len(data['cc'][i]))