from matplotlib import pyplot as plt, rcParams
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import auc, roc_curve
from Utils import pickle_save, pickle_load
from collections import Counter
import math
import CONSTANTS
import obonet
import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



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



def plot_auc_distribution(data):

    keys = {"mf": 0, "cc": 1, "bp": 2}
    titles = {"mf": "Molecular Function", "cc": "Cellular Component", "bp": "Biological Process"}

    nrows, ncols = 3, 1
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(32, 28),
        gridspec_kw={'hspace': 0.4}
    )

    for i, j in data.items():
        Xs = ['_'.join(map(str, s[0])) for s in j]
        Ys = [s[3] for s in j]
        Cs = [s[2] for s in j]
        Cs1 = ['\n\n\n {}'.format(s[1]) for s in j]
        bars = axs[keys[i]].bar(Xs, Ys, label="AUC");
        axs[keys[i]].bar_label(bars, labels=Cs, fontsize=24, fontweight="bold", color='black', label_type='center')
        axs[keys[i]].bar_label(bars, labels=Cs1, fontsize=24, fontweight="bold", color='navy', label_type='edge')
        axs[keys[i]].set_title(titles[i], fontsize=24, fontweight="bold")
        axs[keys[i]].tick_params(axis='both', labelrotation=25, labelsize=20)
        axs[keys[i]].set_xlabel('Annotation size',  fontweight ='bold', fontsize=24)
        axs[keys[i]].set_ylabel('Average AUC',  fontweight ='bold', fontsize=24)
        
    plt.suptitle(f'Average AUCs of terms grouped by annotation size on the test dataset\n', fontsize=26, ha='center', fontweight="bold")
    plt.tight_layout()   
    plt.savefig("plots/average_auc.png")


 
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


def compute_auc():
    results = {'cc':[], 'mf':[], 'bp':[]}
    test_group = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
    test_groundtruth = load_predictions(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")
    to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
    thresholds = [(1, 5), (6, 10), (11, 15),(16, 20), (21, 25), (26, 30), 
                  (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60), 
                  (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90),
                  (91, 95), (96, 100)]

    onts = ['cc', 'mf', 'bp']

    all_terms = set()
    test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")
    for i in test_groundtruth:
        all_terms.update(test_groundtruth[i])
    
    for ont in onts:
        for threshold in thresholds:
            rare_terms = {term: pos for pos, term in enumerate(get_rare_terms(ont, threshold[0], threshold[1]))}
            # print(ont, threshold, len(rare_terms))
            # get all predictions for particular method
            pth = "evaluation/results/{}_out".format("full_linear")
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
                _pred = set([i[0] for i in predictions[protein]]).intersection(rare_terms)
                _labs = set([i for i in test_groundtruth[protein]]).intersection(rare_terms)

                for _pos, t in enumerate(rare_terms_list):
                    if t in _pred:
                        preds[pos, _pos] = 1.0
                    if t in _labs:
                        labels[pos, _pos] = 1.0


            count = 0
            total = 0
            for _pos, t in enumerate(rare_terms_list):
                pos_n = np.sum(labels[:, _pos])
                if pos_n > 0:
                    roc_auc, fpr, tpr = compute_roc(labels[:, _pos], preds[:, _pos])
                    # print(t, roc_auc)
                    total += roc_auc
                    count += 1
            print(f'Average AUC for {ont} {threshold} {len(rare_terms)} {len(all_terms.intersection(rare_terms))} {total / count:.3f}')
            results[ont].append((threshold, len(rare_terms), len(all_terms.intersection(rare_terms)), round(total/count, 3)))

    # print(results)   
    plot_auc_distribution(results)         
    
        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr


    

         
check_intersection_rare()
compute_auc()
            
