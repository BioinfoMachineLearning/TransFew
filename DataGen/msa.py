# HHblits msa generation
import sys
import csv
import subprocess
from os import listdir
from Bio import SeqIO

def generate_msas():
    # this part was run on lotus
    gen_a3ms = listdir("/bmlfast/frimpong/shared_function_data/a3ms/")
    gen_a3ms = set([i.split(".")[0] for i in gen_a3ms])
    
    single_fasta  = listdir("/bmlfast/frimpong/shared_function_data/single_fastas/")
    single_fasta = set([i.split(".")[0] for i in single_fasta])

    all_fastas = list(single_fasta.difference(gen_a3ms))
    print(len(single_fasta), len(gen_a3ms), len(all_fastas))

    
    all_fastas.sort()

    for fasta in all_fastas[0:100]:
        CMD = "hhblits -i /bmlfast/frimpong/shared_function_data/single_fastas/{}.fasta \
            -d /bmlfast/frimpong/msa_database/uniref30/UniRef30_2023_02 -oa3m \
            /bmlfast/frimpong/shared_function_data/a3ms/{}.a3m -cpu 2 -n 2".format(fasta, fasta)
        subprocess.call(CMD, shell=True, cwd="/bmlfast/frimpong/shared_function_data/")




generate_msas()

