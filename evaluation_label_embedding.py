from matplotlib import pyplot as plt, rcParams
import numpy as np
from Utils import pickle_save, pickle_load
from collections import Counter
import math
import CONSTANTS
import obonet
import networkx as nx


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
    
    methods = ["full_x", "full_biobert", "full_gcn", "full_linear"]
    

    colors = ["grey", "orange", "orangered", "indigo", "blue", "red", "darkgreen", "darkblue", "gold", "azure", " black"]

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


    # plot_curves(dic, ontology=ontology)



go_graph = get_graph()

test_group = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
test_groundtruth = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/groundtruth")



print(type(test_group['LK_bp']))
# add limited known and no known
test_group = {
    'bp': test_group['LK_bp'] | test_group['NK_bp'],
    'mf': test_group['LK_mf'] | test_group['NK_mf'],
    'cc': test_group['LK_cc'] | test_group['NK_cc']
}



titles = {
    "cc": "Cellular Component",
    "mf": "Molecular Function",
    "bp": "Biological Process",
}
to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}

def main():
    for ont in test_group:

        #if group == "LK_bp" or group == "NK_bp":# or group == "NK_cc" or group == "LK_cc":
         #   continue
    
        print("###############{}######################".format(ont))

        proteins = set(test_group[ont]).difference(to_remove)

        print("Evaluating:{}, total number of proteins:{}".format(ont, len(proteins)))

        groundtruth = {i: test_groundtruth[i] for i in proteins}
        title = titles[ont]

        evaluating(proteins=proteins, groundtruth=groundtruth, go_graph=go_graph, title=title, ontology=ont)
        
        print("#####################################")


if __name__ == '__main__':
    main()