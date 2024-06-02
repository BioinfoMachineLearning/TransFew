import subprocess
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from Utils import is_file, load_ckp, pickle_load
import os
import argparse
from models.model import TFun
from Dataset.MyDataset import PredictDataset

def write_output(results, terms, filepath, cutoff=0.001):
    with open(filepath, 'w') as fp:
        for prt in results:
            assert len(terms) == len(results[prt])
            tmp = list(zip(terms, results[prt]))
            tmp.sort(key = lambda x: x[1], reverse=True)
            for trm, score in tmp:
                if score > cutoff:
                    fp.write('%s\t%s\t%0.3f\n' %  (prt, trm, score))


def generate_bulk_embedding(wd, fasta_path):
    # name model output dir, embedding layer 1, embedding layer 2, batch
    model = ("esm_2", "esm2_t48_15B_UR50D", fasta_path, "esm2_t48", 48, 100)
    CMD = "python -u {} {} {} {}/{} --repr_layers {} --include mean per_tok " \
                    "--toks_per_batch {}".format("external/extract.py", model[1], model[2], \
                                                wd, model[3], model[4], model[5])
    print(CMD)
    subprocess.call(CMD, shell=True, cwd="./")


def fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def merge_pts(keys, fasta, wd):
    for pos, protein in enumerate(keys):
        fasta_dic = fasta_to_dictionary(fasta)

        tmp = []
        for level in range(keys[protein]):
            os_path = "{}/esm2_t48/{}_{}.pt".format(wd, protein, level)
            tmp.append(torch.load(os_path))

        data = {'representations': {}, 'mean_representations': {}}
        for index in tmp:
            # print(index['mean_representations'][rep].shape, torch.mean(index['representations'][rep], dim=0).shape)
            assert torch.equal(index['mean_representations'][48], torch.mean(index['representations'][48], dim=0))

            if 48 in data['representations']:
                data['representations'][48] = torch.cat((data['representations'][48], index['representations'][48]))
            else:
                data['representations'][48] = index['representations'][48]

        assert len(fasta_dic[protein][3]) == data['representations'][48].shape[0]

        data['mean_representations'][48] = torch.mean(data['representations'][48], dim=0)

        # print("saving {}".format(protein))
        torch.save(data, "{}/esm2_t48/{}.pt".format(wd, protein))


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record

def crop_fasta(record):
    splits = []
    keys = {}
    main_id = record.id
    chnks = len(record.seq) / 1021
    remnder = len(record.seq) % 1021
    chnks = int(chnks) if remnder == 0 else int(chnks) + 1
    keys[main_id] = chnks
    for pos in range(chnks):
        id = "{}_{}".format(main_id, pos)
        seq = str(record.seq[pos * 1021:(pos * 1021) + 1021])
        splits.append(create_seqrecord(id=id, name=id, description="", seq=seq))
    return splits, keys



def generate_embeddings(in_fasta, wd):
    keys = {}
    sequences = []
    proteins = []
    input_seq_iterator = SeqIO.parse(in_fasta, "fasta")
    for record in input_seq_iterator:
        proteins.append(record.id)
        if len(record.seq) > 1021:
            _seqs, _keys = crop_fasta(record)
            sequences.extend(_seqs)
            keys.update(_keys)
        else:
            sequences.append(record)

    # any sequence > 1022
    cropped_fasta = wd + "/temp.fasta"
    if len(keys) > 0:
        SeqIO.write(sequences, cropped_fasta, "fasta")
        generate_bulk_embedding(wd, cropped_fasta)
        merge_pts(keys, fasta_path, wd)
    else:
        generate_bulk_embedding(wd, in_fasta)

    return proteins


def create_dataset(proteins, wd):
    data = {'esm2_t48': [], 'protein': [] }
    for _protein in proteins:
        tmp = torch.load("{}/esm2_t48/{}.pt".format(wd, _protein))
        tmp = tmp['mean_representations'][48].view(1, -1).squeeze(0).cpu()

        data['esm2_t48'].append(tmp)
        data['protein'].append(_protein)

    dataset = PredictDataset(data=data)
    return dataset


def get_term_indicies(ontology, device, data_path):

    _term_indicies = pickle_load(data_path + "/{}/term_indicies".format(ontology))

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


parser = argparse.ArgumentParser(description=" Predict protein functions with TransFew ", epilog=" Thank you !!!")
parser.add_argument('--data-path', type=str, default="", help="Path to data files (models)")
parser.add_argument('--working-dir', type=str, default=".", help="Path to generate temporary files")
parser.add_argument('--ontology', type=str, default="cc", help="Path to data files")
parser.add_argument('--no-cuda', default=False, help='Disables CUDA training.')
parser.add_argument('--batch-size', default=10, help='Batch size.')
parser.add_argument('--fasta-path', default="sequence.fasta", help='Path to Fasta')
parser.add_argument('--output', type=str, default="result.tsv", help="File to save output")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = 'cuda:1'
else:
    device = 'cpu'


fasta_path = args.fasta_path
wd = args.working_dir
ontology = args.ontology
data_path = args.data_path


proteins = generate_embeddings(in_fasta=fasta_path, wd=wd)

dataset = create_dataset(proteins, wd=wd)

loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)


sorted_terms = pickle_load(data_path+"/{}/sorted_terms".format(ontology))
full_term_indicies, freq_term_indicies, rare_term_indicies, rare_term_indicies_2 = \
    get_term_indicies(ontology=ontology, device=device, data_path=data_path)

kwargs = {
    'device': device,
    'ont': ontology,
    'full_indicies': full_term_indicies,
    'freq_indicies': freq_term_indicies,
    'rare_indicies': rare_term_indicies,
    'rare_indicies_2': rare_term_indicies_2,
    'sub_model': 'full',
    'load_weights': True,
    'label_features': 'gcn',
    'group': ""
}

ckp_dir = data_path + '/{}/models/{}_{}/'.format(ontology, kwargs['sub_model'], kwargs['label_features'])
ckp_pth = ckp_dir + "current_checkpoint.pt"
model = TFun(**kwargs)

# load model
model = load_ckp(checkpoint_dir=ckp_dir, model=model, best_model=False, model_only=True)

model.to(device)
model.eval()

results = {}
for data in loader:
    _features, _proteins = data[:1], data[1]
    output = model(_features)
    output = torch.index_select(output, 1, full_term_indicies)
    output = output.tolist()

    for i, j in zip(_proteins, output):
        results[i] = j

terms = [sorted_terms[i] for i in full_term_indicies]
write_output(results, terms, "{}/{}".format(wd, args.output), cutoff=0.01)

            