import torch
from Utils import create_directory, load_ckp, pickle_load, pickle_save
import CONSTANTS
import math, os, time
import argparse
from models.model import TFun, TFun_submodel
from Dataset.MyDataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--load_weights", default=False, type=bool, help='Load weights from saved model')
args = parser.parse_args()

torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = 'cuda:1'
else:
    device = 'cpu'

# load all test
all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/output_t1_t2/test_proteins")


ontologies = ["cc", "mf", "bp"]
models = ['esm2_t48', 'msa_1b', 'interpro', 'full']



def write_output(results, terms, filepath, cutoff=0.001):
    with open(filepath, 'w') as fp:
        for prt in results:
            assert len(terms) == len(results[prt])
            tmp = list(zip(terms, results[prt]))
            tmp.sort(key = lambda x: x[1], reverse=True)
            for trm, score in tmp:
                if score > cutoff:
                    fp.write('%s\t%s\t%0.3f\n' %  (prt, trm, score))



def get_term_indicies(ontology):

    _term_indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/term_indicies".format(ontology))

    if ontology == 'bp':
        full_term_indicies, mid_term_indicies,  freq_term_indicies =  _term_indicies[0], _term_indicies[5], _term_indicies[30]
        rare_term_indicies_2 = torch.tensor([i for i in full_term_indicies if not i in set(mid_term_indicies)]).to(device)
        rare_term_indicies = torch.tensor([i for i in mid_term_indicies if not i in set(freq_term_indicies)]).to(device)
        full_term_indicies, freq_term_indicies =  torch.tensor(_term_indicies[0]).to(device), torch.tensor(freq_term_indicies).to(device)
    else:
        full_term_indicies =  _term_indicies[0]
        freq_term_indicies = _term_indicies[30]
        rare_term_indicies = torch.tensor([i for i in full_term_indicies if not i in set(freq_term_indicies)]).to(device)
        full_term_indicies =  torch.tensor(full_term_indicies).to(device)
        freq_term_indicies = torch.tensor(freq_term_indicies).to(device)
        rare_term_indicies_2 = None

    return full_term_indicies, freq_term_indicies, rare_term_indicies, rare_term_indicies_2



for ontology in ontologies:

    data_pth = CONSTANTS.ROOT_DIR + "test/dataset/{}".format(ontology)
    sorted_terms = pickle_load(CONSTANTS.ROOT_DIR+"/{}/sorted_terms".format(ontology))
    
    for sub_model in models:

        tst_dataset = TestDataset(data_pth=data_pth, submodel=sub_model)
        tstloader = torch.utils.data.DataLoader(tst_dataset, batch_size=500, shuffle=False)
        # terms, term_indicies, sub_indicies = get_term_indicies(ontology=ontology, submodel=sub_model)
        full_term_indicies, freq_term_indicies, rare_term_indicies, rare_term_indicies_2 = get_term_indicies(ontology=ontology)


        kwargs = {
            'device': device,
            'ont': ontology,
            'full_indicies': full_term_indicies,
            'freq_indicies': freq_term_indicies,
            'rare_indicies': rare_term_indicies,
            'rare_indicies_2': rare_term_indicies_2,
            'sub_model': sub_model,
            'load_weights': True,
            'group': ""
        }

        if sub_model == "full":
            print("Generating for {} {}".format(ontology, sub_model))

            ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}_gcn_old/'.format(ontology, sub_model)
            ckp_pth = ckp_dir + "current_checkpoint.pt"
            model = TFun(**kwargs)

            # load model
            model = load_ckp(checkpoint_dir=ckp_dir, model=model, best_model=False, model_only=True)

            model.to(device)
            model.eval()

            results = {}
            for data in tstloader:
                _features, _proteins = data[:4], data[4]
                output = model(_features)
                output = torch.index_select(output, 1, full_term_indicies)
                output = output.tolist()

                for i, j in zip(_proteins, output):
                    results[i] = j

            terms = [sorted_terms[i] for i in full_term_indicies]


            filepath = CONSTANTS.ROOT_DIR + 'evaluation/raw_predictions/transfew/'
            create_directory(filepath)
            write_output(results, terms, filepath+'{}.tsv'.format(ontology), cutoff=0.01)

        else:
            print("Generating for {} {}".format(ontology, sub_model))

            ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}/'.format(ontology, sub_model)
            ckp_pth = ckp_dir + "current_checkpoint.pt"
            
            model = TFun_submodel(**kwargs)
            model.to(device)

            # print("Loading model checkpoint @ {}".format(ckp_pth))
            model = load_ckp(checkpoint_dir=ckp_dir, model=model, best_model=False, model_only=True)
            model.eval()
            
            results = {}
            for data in tstloader:
                _features, _proteins = data[0], data[1]

                output = model(_features).tolist()
                for i, j in zip(_proteins, output):
                    results[i] = j

            terms = [sorted_terms[i] for i in freq_term_indicies]

            filepath = CONSTANTS.ROOT_DIR + 'evaluation/raw_predictions/{}/'.format(sub_model)
            create_directory(filepath)
            write_output(results, terms, filepath+'{}.tsv'.format(ontology), cutoff=0.01)

                



