import os
import subprocess
import networkx as nx
import numpy as np
import obonet
import pandas as pd
from Bio import SeqIO
import pickle

ROOT_DIR = "/home/fbqc9/Workspace/DATA/"


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)
    

def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def get_graph(go_path=ROOT_DIR + "obo/go-basic.obo"):
    go_graph = obonet.read_obo(open(go_path, 'r'))

    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)
    go_graph.remove_edges_from(unaccepted_edges)

    return go_graph


def create_db(wd, fasta_seq, dbase_name):
    command = "diamond makedb --in {} -d {}".format(fasta_seq, dbase_name)
    subprocess.call(command, shell=True, cwd="{}".format(wd))


def diamomd_blast(wd, dbase_name, query, output):
    command = "diamond blastp -d {} -q {} --outfmt 6 qseqid sseqid bitscore -o {}". \
        format(dbase_name, query, output)
    # command = "diamond blastp -d {} -q {}  -o {} --sensitive" \
      #     .format(dbase_name, query, output)

    subprocess.call(command, shell=True, cwd="{}".format(wd))


def create_diamond(proteins, groundtruth, diamond_scores_file):

    train_test_proteins = set(ontology_groundtruth.keys())
    proteins = set(proteins)

    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] in proteins and it[1] in train_test_proteins:
                if it[0] not in diamond_scores:
                    diamond_scores[it[0]] = {}
                diamond_scores[it[0]][it[1]] = float(it[2])

    # BlastKNN
    results = {}
    for protein in proteins:
        tmp_annots = []
        if protein in diamond_scores:
            sim_prots = diamond_scores[protein]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= groundtruth[p_id]
                total_score += score
            # allgos is all go terms for protein
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)

            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in groundtruth[p_id]:
                        s = max(s, score)
                sim[j] = s
            sim = sim / np.max(sim)
            for go_id, score in zip(allgos, sim):
                tmp_annots.append((go_id, score))
        results[protein] = tmp_annots

    return results


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


train_fasta = ROOT_DIR + "uniprot/train_sequences.fasta"
wd = ROOT_DIR + "evaluation/raw_predictions/diamond"
dbase_name = "diamond_db"
query = wd + "/test_fasta"
diamond_res = wd + "/diamond_res"

print(count_proteins(train_fasta))
# create_db(wd, train_fasta, dbase_name)
# diamomd_blast(wd, dbase_name, query, diamond_res)

go_graph = get_graph()
test_group = pickle_load("/home/fbqc9/Workspace/DATA/test/output_t1_t2/test_proteins")
groundtruth = pickle_load("/home/fbqc9/Workspace/DATA/groundtruth")
parent_terms = {
    'cc': 'GO:0005575',
    'mf': 'GO:0003674',
    'bp': 'GO:0008150'
}

for ont in test_group:
    parent_term = parent_terms[ont]

    train_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/train_proteins".format(ont)))
    valid_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/validation_proteins".format(ont)))
    data = train_data + valid_data

    all_go_terms = nx.ancestors(go_graph, parent_term)#.union(set([parent_term]))
    ontology_groundtruth = {prot: set(groundtruth[prot]).intersection(all_go_terms) for prot in data}


    for sptr in test_group[ont]:

        print("Swissprot or Trembl is {}".format(sptr))

        dir_pth = ROOT_DIR + "evaluation/predictions/{}_{}/".format(sptr, ont)
        create_directory(dir_pth)

        proteins = test_group[ont][sptr]

        diamond_scores = create_diamond(proteins, ontology_groundtruth, diamond_res)

        file_out = open(dir_pth+"{}.tsv".format("diamond"), 'w')
        for prot in proteins:
            for annot in diamond_scores[prot]:
                file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
        file_out.close()
