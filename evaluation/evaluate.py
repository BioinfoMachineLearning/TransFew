"""
This script is used for evaluation.
Script is taken from deepgraphgo
"""

import pickle

import networkx as nx
import numpy as np
import obonet
import pandas as pd
import Constants
from preprocessing.utils import pickle_load, read_test_set, pickle_save
from matplotlib import pyplot as plt, rcParams





def get_pred(proteins, labels, my_preds, go_graph, go_set):
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    for t in range(0, 101):
        threshold = t / 100.0
        preds = []
        labs = []
        for protein in proteins:
            labs.append(labels[protein])
            annots = set()

            for go_id, score in my_preds[protein].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                if go_id in go_graph:
                    new_annots |= nx.descendants(go_graph, go_id) | {go_id}
            new_annots = new_annots.difference(Constants.root_terms)
            preds.append(new_annots.intersection(go_set))

        fscore, prec, rec = evaluate_annotations(labs, preds)

        precisions.append(prec)
        recalls.append(rec)
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            fmax_pos = (rec, prec)
    return precisions, recalls, fmax, fmax_pos, tmax


def evaluate(ontology, obo, proteins, ground_truth, title, fig_name):
    go_graph = obonet.read_obo(open(obo, 'r'))
    go_set = nx.ancestors(go_graph, Constants.FUNC_DICT[ontology])

    labels = {protein: go_set.intersection(ground_truth[protein]) for protein in proteins}

    dic = {
        "Naive": {'color': 'silver'},
        "Diamond": {'color': 'orange'},
        "DeepGOCNN": {'color': 'green'},
        "Tale": {'color': 'blue'},
        "TransFun": {'color': 'red'},
        "DeepGOPlus": {'color': 'darkgreen'},
        "Tale+": {'color': 'darkblue'},
        "TransFun+": {'color': 'darkred'},
        "DeepFri": {'color': 'gold'}
    }

    # dic = {
    #     "TransFun+": {'color': 'darkblue'}
    # }
    methods = ["Naive", "Diamond", "DeepGOCNN", "Tale", "TransFun", "DeepGOPlus", "Tale+", "TransFun+", "DeepFri"]
    #methods = ["TransFun+"]

    for method in methods:
        print("Computing method: {} Ontology: {}".format(method, ontology))
        predictions = pickle_load(
            Constants.ROOT + "{}/timebased/{}".format(method, Constants.NAMESPACES[ontology]))

        coverage = len(set(predictions.keys()).intersection(proteins)) / len(proteins)

        pred_proteins = set(predictions.keys())

        _labels = {key: value for key, value in labels.items() if key in pred_proteins}
        _proteins = set(proteins).intersection(pred_proteins)
        precisions, recalls, fmax, fmax_pos, tmax = get_pred(_proteins, _labels, predictions, go_graph, go_set)

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
        print(f'Fmax: {fmax:0.3f}, threshold: {tmax}, Coverage: {coverage}, AUPR: {aupr:0.3f}')

    return dic
    # plot_curves(dic, Constants.NAMES[ontology], title=title, fig_name=fig_name)
    #
    # df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    # df.to_pickle(Constants.ROOT + "eval/Results/PR_{}_{}.pkl".format(ontology, seq_id))


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
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)

        tpn = len(tp)
        total += 1
        recall = tpn / (1.0 * (tpn + fn))
        total_recall += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fp))
            total_precision += precision
    total_recall /= total
    if p_total > 0:
        total_precision /= p_total
    f_1 = 0.0
    if total_precision + total_recall > 0:
        f_1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    return f_1, total_precision, total_recall


ontologies = Constants.NAMESPACES
obo = Constants.ROOT + "obo/go.obo"


# proteins = pickle_load(Constants.ROOT + '0.3/validation')


def plot_curves(data, ax, ontology):
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
                label=f'{method}: Coverage {coverage: 0.2f}, fmax {fmax: 0.2f}, AUPR {aupr:0.2f})')
        ax.plot(recalls[fmax_pos], precisions[fmax_pos], 'ro', color='black')
        ax.scatter(recalls[fmax_pos], precisions[fmax_pos], s=rcParams['lines.markersize'] ** 3, facecolors='none',
                   edgecolors='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('{}'.format(ontology))
    ax.legend(loc="upper right")



def evaluating(eval_filter={}, title="", fig_name=""):

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Area Under the Precision-Recall curve")
    for pos, ontology in enumerate(ontologies):

        data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")
        data = data.loc[data['ONTOLOGY'] == Constants.NAMESPACES[ontology]]
        # print(data_bp.columns)

        for key in eval_filter:
            data = data.loc[data[key] == eval_filter[key]]

        missing = set(pickle_load(Constants.ROOT + "timebased/missing_proteins"))
        data = list(set(data['ACC'].to_list()).difference(missing))

        ground_truth = pickle_load(Constants.ROOT + "timebased/true_labels")
        results = evaluate(ontology=ontology, obo=obo, proteins=data, ground_truth=ground_truth, title=title, fig_name=fig_name)

        plot_curves(results, ax=ax[pos], ontology=ontology)

    plt.show()
    exit()
    plt.savefig(Constants.ROOT + "timebased/plots/aupr_{}.png".format(fig_name))

def evaluating_length(title="Sequence > 1000", fig_name=""):
    for ontology in ontologies:
        data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")
        data = data.loc[data['ONTOLOGY'] == Constants.NAMESPACES[ontology]]

        data = data.loc[data['SEQUENCE LENGTH'] >= 1024]
        # print(data_bp.shape)

        missing = set(pickle_load(Constants.ROOT + "timebased/missing_proteins"))
        data = list(set(data['ACC'].to_list()).difference(missing))

        print(len(data))

        ground_truth = pickle_load(Constants.ROOT + "timebased/true_labels")
        evaluate(ontology=ontology, obo=obo, proteins=data, ground_truth=ground_truth, title=title, fig_name="")


def evaluating_by_seq(title="Sequence ID < 0.3", fig_name=""):
    columns = ["queryID", "targetID", "alnScore", "seqIdentity", "eVal", "qStart", "qEnd", "qLen", "tStart", "tEnd",
               "tLen"]
    seqs = pd.read_csv("../Analysis/mmseq/max/final_output.tsv", sep="\t", names=columns)
    seqs = seqs[["queryID", "targetID", "seqIdentity"]]
    seqs = seqs.set_index('queryID').to_dict()['seqIdentity']

    seqs = {key: value for key, value in seqs.items() if value >= 0.3}

    for ontology in ontologies:
        data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")
        data = data.loc[data['ONTOLOGY'] == Constants.NAMESPACES[ontology]]

        data = data[~data['ACC'].isin(seqs)]

        missing = set(pickle_load(Constants.ROOT + "timebased/missing_proteins"))
        data = list(set(data['ACC'].to_list()).difference(missing))

        print(len(data))

        ground_truth = pickle_load(Constants.ROOT + "timebased/true_labels")
        evaluate(ontology=ontology, obo=obo, proteins=data, ground_truth=ground_truth, title=title, fig_name="")


# All evaluation
# evaluating(fig_name="all")
# #
# # Humans
# evaluating({'TAXONOMY': 9606}, title="Human Species", fig_name="human")
# #
# # # Mouse
# evaluating({'TAXONOMY': 10090}, title="Mouse Species", fig_name="mouse")

# > 1024
# evaluating_length(fig_name="1024")

# # seq < 30
# evaluating_by_seq(fig_name="seq_l_30")
