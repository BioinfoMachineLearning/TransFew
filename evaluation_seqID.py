import os
import shutil
import subprocess
from matplotlib import pyplot as plt, rcParams
import numpy as np
from Utils import count_proteins, create_directory, pickle_save, pickle_load
from collections import Counter
import math
import CONSTANTS
import obonet
import networkx as nx
from Bio import SeqIO


'''
    compute ic
    script adapated from deepgozero
    annotations = train+valid+test annotations
'''
def compute_ics(ontology, go_graph):
    parent_term = CONSTANTS.FUNC_DICT[ontology]
    ont_terms = nx.ancestors(go_graph, parent_term).union(set([parent_term]))

    # get ics and ic norms
    groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "groundtruth")
    train_valid_proteins = set(pickle_load(CONSTANTS.ROOT_DIR + "{}/all_proteins".format(ontology)))
    train_valid_annots = [set(groundtruth[protein]).intersection(ont_terms) for protein in groundtruth if protein in train_valid_proteins]

    cnt = Counter()
    for x in train_valid_annots:
        cnt.update(x)
    ics = {}
    ic_norm = 0.0
    for go_id, n in cnt.items():
        parents = list(go_graph.neighbors(go_id))
        if len(parents) == 0:
            min_n = n
        else:
            min_n = min([cnt[x] for x in parents])

        ics[go_id] = math.log(min_n / n, 2)
        ic_norm = max(ic_norm, ics[go_id])

    return ics, ic_norm


def get_graph(go_path=CONSTANTS.ROOT_DIR + "/obo/go-basic.obo"):
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


def plot_curves(data, ontology):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # fig.suptitle("Area Under the Precision-Recall curve")
    method_dic = {
        "esm2_t48": "ESM", "msa_1b": "MSA", 
        "interpro": "Interpro", "tale": "Tale", 
        "netgo": "NetGO3", "full_x": "FULL Hierachical", 
        "full_biobert": "Full Biobert", "full_gcn": "Full GCN", 
        "full_linear": "Full Linear", "naive": "Naive", 
    }

    for method in data:
        recalls = data[method]["recalls"]
        precisions = data[method]["precisions"]
        fmax_pos = data[method]["fmax_pos"]
        aupr = data[method]["aupr"]
        color = data[method]["color"]
        coverage = data[method]["coverage"]
        fmax = data[method]["fmax"]

        for i in range(len(precisions)):
            if recalls[i] == fmax_pos[0] and precisions[i] == fmax_pos[1]:
                fmax_pos = i
                break

        ax.plot(recalls[1:], precisions[1:], color=color,
                label=f'{method_dic[method]}: Coverage {coverage: 0.2f}, fmax {fmax: 0.2f}, AUPR {aupr:0.2f})')
        ax.plot(recalls[fmax_pos], precisions[fmax_pos], 'ro')#, color='black')
        ax.scatter(recalls[fmax_pos], precisions[fmax_pos], s=rcParams['lines.markersize'] ** 3, facecolors='none',
                   edgecolors='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title("Area Under the Precision-Recall curve -- {}".format(ontology))
    ax.legend(loc="upper right")
    plt.savefig("plots/results_{}.png".format(ontology))


def evaluate_annotations(real_annots, pred_annots, ics, ic_norm):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []

    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i].difference(tp)
        fn = real_annots[i].difference(tp)
        tpic = 0.0
        for go_id in tp:
            ic = ics.get(go_id, 0.0)
            tpic += ic/ic_norm
            avg_ic += ic
        fpic = 0.0
        for go_id in fp:
            ic = ics.get(go_id, 0.0)
            fpic += ic/ic_norm
            mi += ic
        fnic = 0.0
        for go_id in fn:
            ic = ics.get(go_id, 0.0)
            fnic += ic/ic_norm
            ru += ic

        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall

        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf


def load_predictions(pth):
    return pickle_load(pth)


def evaluate_method(proteins, labels, predictions, go_graph, go_set, ics, ic_norm):
    fmax = 0.0
    tmax = 0.0
    wfmax = 0.0
    wtmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []

    for t in range(0, 101):
        threshold = t / 100.0
        preds = []
        labs = []
        for protein in proteins:
            # get labels for ontology
            labs.append(labels[protein].intersection(go_set).difference(CONSTANTS.root_terms))
            annots = set()

            for go_id, score in predictions[protein]:
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                if go_id in go_graph:
                    new_annots |= nx.descendants(go_graph, go_id) | {go_id}
            new_annots = new_annots.difference(CONSTANTS.root_terms)
            preds.append(new_annots.intersection(go_set))

        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(labs, preds, ics=ics, ic_norm=ic_norm)

        precisions.append(prec)
        recalls.append(rec)
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            fmax_pos = (rec, prec)
            avgic = avg_ic
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s
    return precisions, recalls, fmax, fmax_pos, tmax, wfmax, wtmax, smin, avgic


def evaluating(proteins, groundtruth, go_graph, title="", ontology=""):

    parent_term = CONSTANTS.FUNC_DICT[ontology]
    ont_terms = nx.ancestors(go_graph, parent_term).union(set([parent_term]))
    ics, ic_norm  = compute_ics(ontology, go_graph)

    methods = ["naive", "tale", "netgo", "full_gcn"]

    colors = ["grey", "orange", "steelblue", "indigo", "blue", "red", "darkgreen", "magenta", "gold", "teal", " black"]

    dic = { method: {'color': colors[pos]} for pos, method in enumerate(methods)}

    for method in methods:

        pth = "evaluation/results/{}_out".format(method)
        predictions = load_predictions(pth)
        
        # predictions = predictions[group]
        #add both NK and LK
        predictions = predictions["LK_"+ontology] | predictions["NK_"+ontology]

        coverage = len(set(predictions.keys()).intersection(proteins)) / len(proteins)

        precisions, recalls, fmax, fmax_pos, tmax, wfmax, wtmax, smin, avgic = evaluate_method(
                                                     proteins=proteins,
                                                     labels=groundtruth, 
                                                     predictions=predictions, 
                                                     go_graph=go_graph, 
                                                     go_set=ont_terms,
                                                     ics=ics, ic_norm=ic_norm)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)

        dic[method]["recalls"] = recalls
        dic[method]["precisions"] = precisions
        dic[method]["fmax_pos"] = fmax_pos
        dic[method]["aupr"] = aupr
        dic[method]["coverage"] = coverage
        dic[method]["fmax"] = fmax
        print(f'Method: {method} Fmax: {fmax:0.3f}, threshold: {tmax}, Coverage: {coverage}, AUPR: {aupr:0.3f} Weighted Fmax: {wfmax:0.3f}, Smin: {smin:0.3f}')
        # print(f'Fmax: {fmax:0.3f}, threshold: {tmax}, AUPR: {aupr:0.3f}')


    #plot_curves(dic, ontology=ontology)
        

def filter_fasta(proteins, infile, outfile):
    seqs = []
    input_seq_iterator = SeqIO.parse(infile, "fasta")

    for pos, record in enumerate(input_seq_iterator):
        if record.id in proteins:
            seqs.append(record)
    SeqIO.write(seqs, outfile, "fasta")


def extract_from_results(infile):
    file = open(infile)
    lines = []
    for _line in file.readlines():
        line = _line.strip("\n").split("\t")
        lines.append((line[0], line[1], line[3]))
    file.close()
    return lines


def get_seq_less(ontology, test_proteins, seq_id=0.3):
    # mmseqs createdb <i:fastaFile1[.gz|.bz2]> ... <i:fastaFileN[.gz|.bz2]>|<i:stdin> <o:sequenceDB> [options]

    full_train_fasta = "/home/fbqc9/Workspace/DATA/uniprot/train_sequences.fasta"
    test_fasta = "/home/fbqc9/Workspace/DATA/uniprot/test_fasta.fasta"

    train_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/train_proteins".format(ontology)))
    valid_data =  list(pickle_load("/home/fbqc9/Workspace/DATA/{}/validation_proteins".format(ontology)))
    train_data = set(train_data + valid_data) # set for fast lookup
    test_proteins = set(test_proteins) # set for fast lookup


    # make temporary directory
    wkdir = "/home/fbqc9/Workspace/TransFun2/evaluation/seqID/{}".format(seq_id)
    create_directory(wkdir)

    target_fasta = wkdir+"/target_fasta"
    query_fasta = wkdir+"/query_fasta"
    filter_fasta(train_data, full_train_fasta, target_fasta)
    filter_fasta(test_proteins, test_fasta, query_fasta)

    assert len(train_data) == count_proteins(target_fasta)
    assert len(test_proteins) == count_proteins(query_fasta)

    print("Creating target Database")
    target_dbase = wkdir+"/target_dbase"
    CMD = "mmseqs createdb {} {}".format(target_fasta, target_dbase)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Creating query Database")
    query_dbase = wkdir+"/query_dbase"
    CMD = "mmseqs createdb {} {}".format(query_fasta, query_dbase)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Mapping very similar sequences")
    result_dbase = wkdir+"/result_dbase"
    CMD = "mmseqs map {} {} {} {} --min-seq-id {}".\
        format(query_dbase, target_dbase, result_dbase, wkdir, seq_id)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    bestResultDB = wkdir+"/bestResultDB"
    CMD = "mmseqs filterdb {} {} --extract-lines 1".format(result_dbase, bestResultDB)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    final_res = wkdir+"/final_res.tsv"
    CMD = "mmseqs createtsv {} {} {} {}".format(query_dbase, target_dbase, bestResultDB, final_res)
    subprocess.call(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    lines = extract_from_results(final_res)

    shutil.rmtree(wkdir)

    querys, targets, seq_ids = zip(*lines)

    querys = set(querys)
    targets = set(targets)

    assert len(train_data.intersection(querys)) == 0
    assert len(test_proteins.intersection(targets)) == 0

    # the proteins with less than X seq identity to the training set
    return test_proteins.difference(querys)



go_graph = get_graph()

test_group = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")

# add limited known and no known
test_group = {
    'bp': test_group['LK_bp'] | test_group['NK_bp'],
    'mf': test_group['LK_mf'] | test_group['NK_mf'],
    'cc': test_group['LK_cc'] | test_group['NK_cc']
}

to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
titles = {
    "cc": "Cellular Component",
    "mf": "Molecular Function",
    "bp": "Biological Process",
}


final_out = {}

def main():
    for ont in test_group:

    
        print("###############{}######################".format(ont))

        proteins = set(test_group[ont]).difference(to_remove)

        proteins = get_seq_less(ontology=ont, test_proteins=proteins)


        print("Evaluating:{}, total number of proteins:{}".format(ont, len(proteins)))

        groundtruth = {i: test_groundtruth[i] for i in proteins}
        title = titles[ont]

        evaluating(proteins=proteins, groundtruth=groundtruth, go_graph=go_graph, title=title, ontology=ont)
        
        print("#####################################")


if __name__ == '__main__':
    main()