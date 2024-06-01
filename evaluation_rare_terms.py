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

    keys = {"mf": 0, "cc": 1, "bp": 2, "swissprot": 0, "trembl": 1}
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process",
              "swissprot": "Swissprot", "trembl": "Trembl"}

    nrows, ncols = 3, 2
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(20, 16),
    )
    
    for ont, dbsubset in results.items():
        for sptr, cnts in dbsubset.items():
            im = axs[keys[ont]][keys[sptr]].bar(cnts.keys(), cnts.values());
            axs[keys[ont]][keys[sptr]].set_title(titles[ont] + " --- " + titles[sptr], fontsize=18, fontweight="bold")
        
    plt.suptitle(f'Distribution of rare terms in test set \n', fontsize=24, ha='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/rare_distribution.png")


def plot_rare_distribution_bar(data, Xs, measure="auc"):

    keys = {"mf": 1, "cc": 2, "bp": 0}
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process"}
    labels = {"tale":"Tale", "netgo3":"NetGO3", "full_gcn":"TransFew", "sprof": "Sprof"}

    Xs = ['_'.join(map(str, s)) for s in Xs]
    x = np.arange(len(Xs))  # the label locations
    width = 0.2  # the width of the bars

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
    fig.text(0.08, 0.5, 'Average AUC', va='center', rotation='vertical', fontweight ='bold', fontsize=16)
    # plt.suptitle(f'Average AUCs of terms grouped by annotation size on the test dataset\n', x=0.5, y=0.9, fontsize=26, ha='center', fontweight="bold")

    plt.tight_layout()  
    plt.savefig("plots/average_new_{}.png".format(measure), bbox_inches='tight')


def plot_rare_distribution_line(data, Xs):
    # got these colors from the cafa-evaluator tool plot assignment
    colors = {
        "deepgose": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        "netgo3": (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        "sprof": (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
        "tale": (1.0, 0.4980392156862745, 0.054901960784313725),
        "transfew": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    }

    keys = {"mf": 1, "cc": 2, "bp": 0}
    ontologies = ["cc", "mf", "bp"]
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process"}

    Xs = ['_'.join(map(str, s)) for s in Xs]
    x = np.arange(len(Xs))

    nrows, ncols = 3, 1
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(14, 12),
        gridspec_kw={'hspace': 0.6},
        sharex=False, sharey=True,
        layout='constrained'
    )

    for ont in ontologies:
        ont_values = data[ont]
        for method, values in ont_values.items():
            axs[keys[ont]].plot(Xs, values, color=colors[method], label=method)
            axs[keys[ont]].plot(Xs, values, color=colors[method], marker='o', markersize=10, mfc='none')
            axs[keys[ont]].plot(Xs, values, color=colors[method], marker='o', markersize=5)


        axs[keys[ont]].set_title(titles[ont], fontsize=18)
        axs[keys[ont]].set_xticks(x, Xs)
        axs[keys[ont]].tick_params(axis='both', labelrotation=45, labelsize=14)
        axs[keys[ont]].legend(loc='upper left', ncols=5, fontsize=16)
    
    fig.text(0.5, 0.04, 'Annotation Frequency', ha='center', fontsize=16)
    fig.text(0.08, 0.5, 'Average AUC', va='center', rotation='vertical', fontsize=16)
    # plt.suptitle(f'Average AUCs of terms grouped by annotation size on the test dataset\n', x=0.5, y=0.9, fontsize=26, ha='center', fontweight="bold")
    
    plt.tight_layout()  
    plt.savefig("plots/average_new.png", bbox_inches='tight')

 
def check_intersection_rare():
    # rare terms
    onts = ["mf", "cc", "bp"]
    db_subset = ['swissprot', 'trembl']
    results = {i: {} for i in onts}
    
    for ont in onts:
        for sptr in db_subset:
            rare_terms = set()
            term_stats = pickle_load(CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
            for term in term_stats:
                if term_stats[term][0] <= 30 :
                        rare_terms.add(term)

            all_terms = set()
            test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruth")[ont]

            proteins = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/test_proteins")[ont][sptr]

            for protein in proteins:
                all_terms.update(test_groundtruth[protein])

            rare_terms_in_test = all_terms.intersection(rare_terms)

            tmp = []
            for i in rare_terms_in_test:
                tmp.append(term_stats[i][0])

            w = collections.Counter(tmp)
            results[ont][sptr] = w

    plot_distribution(results)


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr


def load_predictions(in_file):
    data = {}
    with open(in_file) as f:
        lines = [line.strip('\n').split("\t") for line in f]

    for line in lines:
        if line[0] in data:
            data[line[0]].append((line[1], line[2]))
        else:
            data[line[0]] = [(line[1], line[2]), ]
    return data


def get_rare_terms(ont, bottom, top):
    
    rare_terms = []
    term_stats = pickle_load(CONSTANTS.ROOT_DIR+"/{}/terms_stats".format(ont))
    for term in term_stats:
        if term_stats[term][0] >= bottom and term_stats[term][0] <= top:
            rare_terms.append(term)
    return rare_terms


def compute_measure():
    onts = ['cc', 'mf', 'bp']
    results = {i: {} for i in onts}
    
    test_proteins = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/test_proteins")
    test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruth")
    
    thresholds = [(1, 5), (6, 10), (11, 15),(16, 20), (21, 25), (26, 30), 
                  (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60), 
                  (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90),
                  (91, 95), (96, 100)]

    methods = ["tale", "netgo3", "sprof", "deepgose", "transfew"]
    db_subset = "swissprot"
    
    for ont in onts:
        for method in methods:
            results[ont][method] = []
            for threshold in thresholds:
                rare_terms = {term: pos for pos, term in enumerate(get_rare_terms(ont, threshold[0], threshold[1]))}
                
                # get all predictions for particular method
                pth = CONSTANTS.ROOT_DIR + "evaluation/predictions/full/{}_{}/{}.tsv".format(db_subset, ont, method)
                predictions = load_predictions(pth)

                # get all proteins in ontology
                proteins = test_proteins[ont][db_subset]

                preds = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)
                labels = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)


                # convert to numpy array
                for pos, protein in enumerate(proteins):
                    _pred = {go_term: score for go_term, score in predictions[protein] if go_term in rare_terms}
                    _labs = test_groundtruth[ont][protein].intersection(rare_terms)

                    for go_id, go_pos in rare_terms.items():
                        if go_id in _pred:
                            preds[pos, go_pos] = _pred[go_id]
                        if go_id in _labs:
                            labels[pos, go_pos] = 1.0

                
                count = 0
                total = 0
                for go_id, go_pos in rare_terms.items():
                    pos_n = np.sum(labels[:, go_pos])
                    if pos_n > 0:
                        # _score = roc_auc_score(labels[:, go_pos], preds[:, go_pos])
                        _score, _, _ = compute_roc(labels[:, go_pos], preds[:, go_pos])
                        total += _score    
                        count += 1
                score = round(total/count, 3)
                results[ont][method].append(score)
            
    pickle_save(results, "rest")    
    pickle_save(thresholds, "thres")    
    # plot_rare_distribution(results, Xs=thresholds, measure=measure)



# Just duplicated code:///
def compute_pearson():
    onts = ['cc', 'mf', 'bp']
    results = {'cc': {}, 'mf': {}, 'bp': {}}
    
    test_proteins = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/test_proteins")
    test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruth")

    thresholds = [(1, 5), (6, 10), (11, 15),(16, 20), (21, 25), (26, 30), 
                  (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60), 
                  (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90),
                  (91, 95), (96, 100)]

    methods = ["tale", "netgo3", "sprof", "deepgose", "transfew"]
    db_subset = "swissprot"
    
    for ont in onts:
        for method in methods:
            print(method)
            results[ont][method] = {'scores': [], 'thresholds': []}
            for op, threshold in enumerate(thresholds):
                rare_terms = {term: pos for pos, term in enumerate(get_rare_terms(ont, threshold[0], threshold[1]))}

                # print(ont, threshold, len(rare_terms))
                pth = CONSTANTS.ROOT_DIR + "evaluation/predictions/full/{}_{}/{}.tsv".format(db_subset, ont, method)
                predictions = load_predictions(pth)

                # get all proteins in ontology
                proteins = test_proteins[ont][db_subset]

                preds = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)
                labels = np.zeros((len(proteins), len(rare_terms)), dtype=np.float32)


                # convert to numpy array
                # convert to numpy array
                for pos, protein in enumerate(proteins):
                    _pred = {go_term: score for go_term, score in predictions[protein] if go_term in rare_terms}
                    _labs = test_groundtruth[ont][protein].intersection(rare_terms)

                    for go_id, go_pos in rare_terms.items():
                        if go_id in _pred:
                            preds[pos, go_pos] = _pred[go_id]
                        if go_id in _labs:
                            labels[pos, go_pos] = 1.0

                count = 0
                total = 0
                for go_id, go_pos in rare_terms.items():
                    pos_n = np.sum(labels[:, go_pos])
                    if pos_n > 0:
                        # _score = roc_auc_score(labels[:, go_pos], preds[:, go_pos])
                        _score, _, _ = compute_roc(labels[:, go_pos], preds[:, go_pos])
                        total += _score    
                        count += 1

                score = round(total/count, 3)
                results[ont][method]['scores'].append(round(score, 3))
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

    

         
# check_intersection_rare()

# compute_measure()
compute_pearson()
