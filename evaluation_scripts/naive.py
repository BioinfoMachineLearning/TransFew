import os
import obonet
import pandas as pd
from collections import Counter
import pickle
import networkx as nx

ROOT_DIR = "/home/fbqc9/Workspace/DATA/"

def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


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


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_naive(groundtruth):

    frequency = []
    for prot, annot in groundtruth.items():
        frequency.extend(list(annot))

    cnt = Counter(frequency)
    max_n = cnt.most_common(1)[0][1]

    scores = []
    for go_id, n in cnt.items():
        score = n / max_n
        scores.append((go_id, score))

    return scores


parent_terms = {
    'cc': 'GO:0005575',
    'mf': 'GO:0003674',
    'bp': 'GO:0008150'
}

go_graph = get_graph()
test_group = pickle_load("/home/fbqc9/Workspace/DATA/test/output_t1_t2/test_proteins")
groundtruth = pickle_load("/home/fbqc9/Workspace/DATA/groundtruth")

for ont in test_group:

    parent_term = parent_terms[ont]
    train_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/train_proteins".format(ont)))
    valid_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/validation_proteins".format(ont)))

    data = train_data + valid_data

    all_go_terms = nx.ancestors(go_graph, parent_term)#.union(set([parent_term]))

    ontology_groundtruth = {prot: set(groundtruth[prot]).intersection(all_go_terms) for prot in data}
    naive_scores = create_naive(ontology_groundtruth)

    for sptr in test_group[ont]:
        print("Swissprot or Trembl is {}".format(sptr))

        dir_pth = ROOT_DIR +"evaluation/predictions/{}_{}/".format(sptr, ont)
        create_directory(dir_pth)

        filt_proteins = test_group[ont][sptr]

        file_out = open(dir_pth+"{}.tsv".format("naive"), 'w')
        for prot in filt_proteins:
            for annot in naive_scores:
                file_out.write(prot + '\t' + annot[0] + '\t' + str(annot[1]) + '\n')
        file_out.close()
