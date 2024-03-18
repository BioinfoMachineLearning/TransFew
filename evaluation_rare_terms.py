from matplotlib import pyplot as plt, rcParams
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score
from Utils import pickle_save, pickle_load
from collections import Counter
import math
import CONSTANTS
import obonet
import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats



def plot_distribution(results):

    keys = {"mf": 0, "cc": 1, "bp": 2}
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process"}

    for i, j in results.items():
        print(i, len(j))

    nrows, ncols = 3, 1
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(20, 16),
    )
    
    for i, j in results.items():
        im = axs[keys[i]].bar(j.keys(), j.values());
        axs[keys[i]].set_title(titles[i], fontsize=18, fontweight="bold")
        
    plt.suptitle(f'Distribution of rare terms in test set \n', fontsize=24, ha='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/rare_distribution.png")


def plot_score_distribution(data, Xs, measure="auc", groundtruth_only=True):

    keys = {"mf": 1, "cc": 2, "bp": 0}
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process"}
    labels = {"tale":"Tale", "netgo":"NetGO3", "full_gcn":"TransFew"}

    Xs = ['_'.join(map(str, s)) for s in Xs]
    x = np.arange(len(Xs))  # the label locations
    width = 0.25  # the width of the bars

    nrows, ncols = 3, 1
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(12, 12),
        gridspec_kw={'hspace': 0.4},
        sharex=False, sharey=True,
        #layout='constrained'
    )

    for ont, ont_values in data.items():
        multiplier = 0
        for method, values in ont_values.items():
            offset = width * multiplier
            bars = axs[keys[ont]].bar(x + offset, values, width, label=labels[method]);
            multiplier += 1

        axs[keys[ont]].set_title(titles[ont], fontsize=18, fontweight="bold")
        axs[keys[ont]].set_xticks(x + width, Xs)
        axs[keys[ont]].tick_params(axis='both', labelrotation=25, labelsize=14)
        axs[keys[ont]].legend(loc='upper left', ncols=3, fontsize=16)
    
    fig.text(0.5, 0.04, 'Annotation Frequency', ha='center', fontweight ='bold', fontsize=16)
    if measure == "auc":
        fig.text(0.08, 0.5, 'Average AUC', va='center', rotation='vertical', fontweight ='bold', fontsize=16)
        # plt.suptitle(f'Average AUCs of terms grouped by annotation size on the test dataset\n', x=0.5, y=0.9, fontsize=26, ha='center', fontweight="bold")
    elif measure == "auprc":
        fig.text(0.04, 0.5, 'Average AUPR', va='center', rotation='vertical',  fontweight ='bold', fontsize=16)
        #plt.suptitle(f'Average AUPRCs of terms grouped by annotation size on the test dataset\n', x=0.5, y=0.95, fontsize=26, ha='center', fontweight="bold")
    
    plt.tight_layout()  
    if groundtruth_only:
        plt.savefig("plots/average_1_{}.png".format(measure), bbox_inches='tight')
    else:
        plt.savefig("plots/average_2_{}.png".format(measure), bbox_inches='tight')

 
def check_intersection_rare():
    # rare terms
    onts = ["mf", "cc", "bp"]
    results = {}
    
    for ont in onts:
        rare_terms = set()
        term_stats = pickle_load(CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
        for term in term_stats:
            if term_stats[term][0] <= 30 :
                    rare_terms.add(term)

        all_terms = set()
        test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")
        for i in test_groundtruth:
            all_terms.update(test_groundtruth[i])

        rare_terms_in_test = all_terms.intersection(rare_terms)

        tmp = []
        for i in rare_terms_in_test:
            tmp.append(term_stats[i][0])

        w = collections.Counter(tmp)

        results[ont] = w

        # print(sum(list(w.values())))

    plot_distribution(results)



def load_predictions(pth):
    return pickle_load(pth)



def get_rare_terms(ont, bottom, top):
    
    rare_terms = []
    term_stats = pickle_load(CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
    for term in term_stats:
        if term_stats[term][0] >= bottom and term_stats[term][0] <= top:
            rare_terms.append(term)
    return rare_terms


def compute_measure(measure="auc", groundtruth_only=True):
    results = {'cc': {}, 'mf': {}, 'bp': {}}
    test_group = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
    test_groundtruth = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")
    to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
    thresholds = [(1, 5), (6, 10), (11, 15),(16, 20), (21, 25), (26, 30), 
                  (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60), 
                  (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90),
                  (91, 95), (96, 100)]

    onts = ['cc', 'mf', 'bp']
    methods = methods = ["tale", "netgo", "full_gcn"]
    
    for ont in onts:
        for method in methods:
            results[ont][method] = []
            for threshold in thresholds:
                rare_terms = {term: pos for pos, term in enumerate(get_rare_terms(ont, threshold[0], threshold[1]))}
                # get all predictions for particular method
                pth = "evaluation/results/{}_out".format(method)
                predictions = load_predictions(pth)
                predictions = predictions["LK_"+ont] | predictions["NK_"+ont]

                # get all proteins in ontology
                proteins = set(test_group["LK_{}".format(ont)]).union(set(test_group["NK_{}".format(ont)])).difference(to_remove)

                preds = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)
                labels = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)

                rare_terms_list = list(rare_terms.keys())
                rare_terms = set(rare_terms.keys())


                # convert to numpy array
                for pos, protein in enumerate(proteins):
                    _pred = {i[0]: i[1] for i in predictions[protein] if i[0] in rare_terms}
                    _labs = set([i for i in test_groundtruth[protein]]).intersection(rare_terms)

                    for _pos, t in enumerate(rare_terms_list):
                        if t in _pred:
                            preds[pos, _pos] = _pred[t]
                        if t in _labs:
                            labels[pos, _pos] = 1.0


                if groundtruth_only:
                    count = 0
                    total = 0
                    for _pos, t in enumerate(rare_terms_list):
                        pos_n = np.sum(labels[:, _pos])
                        if pos_n > 0:
                            if measure == "auc":
                                # roc_auc, fpr, tpr = compute_roc(labels[:, _pos], preds[:, _pos])
                                _score = roc_auc_score(labels[:, _pos], preds[:, _pos])
                            elif measure == "auprc":
                                _score = average_precision_score(labels[:, _pos], preds[:, _pos])
                            total += _score    
                            count += 1
                    score = round(total/count, 3)
                else:
                    if measure == "auc":
                        score = roc_auc_score(labels.flatten(), preds.flatten())
                    elif measure == "auprc":
                        score = average_precision_score(labels.flatten(), preds.flatten())
                results[ont][method].append(round(score, 3))
            
    plot_score_distribution(results, Xs=thresholds, measure=measure, groundtruth_only=groundtruth_only)



# Just duplicated code:///
def compute_pearson():
    results = {'cc': {}, 'mf': {}, 'bp': {}}
    test_group = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
    test_groundtruth = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")
    to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}

    thresholds = [(1, 5), (6, 10), (11, 15),(16, 20), (21, 25), (26, 30), 
                  (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60), 
                  (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90),
                  (91, 95), (96, 100)]

    onts = ['cc', 'mf', 'bp']
    methods = methods = ["tale", "netgo", "full_gcn"]
    
    for ont in onts:
        for method in methods:
            print(method)
            results[ont][method] = {'scores': [], 'thresholds': []}
            for op, threshold in enumerate(thresholds):
                rare_terms_list = get_rare_terms(ont, threshold[0], threshold[1])
                if len(rare_terms_list) > 0:
                    rare_terms = {term: pos for pos, term in enumerate(rare_terms_list)}
                    # print(ont, threshold, len(rare_terms))
                    # get all predictions for particular method
                    pth = "evaluation/results/{}_out".format(method)
                    predictions = load_predictions(pth)
                    predictions = predictions["LK_"+ont] | predictions["NK_"+ont]

                    # get all proteins in ontology
                    proteins = set(test_group["LK_{}".format(ont)]).union(set(test_group["NK_{}".format(ont)])).difference(to_remove)

                    preds = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)
                    labels = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)


                    # convert to numpy array
                    for pos, protein in enumerate(proteins):

                        _pred = {i[0]: i[1] for i in predictions[protein] if i[0] in rare_terms}
                        _labs = set([i for i in test_groundtruth[protein]]).intersection(rare_terms)

                        for _pos, t in enumerate(rare_terms_list):
                            if t in _pred:
                                preds[pos, _pos] = _pred[t]
                            if t in _labs:
                                labels[pos, _pos] = 1.0

                    auprc = average_precision_score(labels.flatten(), preds.flatten())
                    results[ont][method]['scores'].append(round(auprc, 3))
                    results[ont][method]['thresholds'].append(op)
                

    
    _compute_pearson(results)


def _compute_pearson(data):
    for ont in data:
        for method in data[ont]:
            print(data[ont][method])
            pearson = stats.pearsonr(data[ont][method]['thresholds'], data[ont][method]['scores'])
            print(ont, method, pearson)
    
        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr

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

def add_anscetor_terms(data):
    go_graph = get_graph()
    new_annots = set()
    for go_id in data:
        if go_id in go_graph:
            new_annots |= nx.descendants(go_graph, go_id) | {go_id}
    return new_annots

    

         
'''check_intersection_rare()
compute_measure(measure="auc", groundtruth_only=True)
compute_measure(measure="auprc", groundtruth_only=True)
compute_measure(measure="auc", groundtruth_only=False)'''
compute_measure(measure="auprc", groundtruth_only=False)
            

# compute_pearson()