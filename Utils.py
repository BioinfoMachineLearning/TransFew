import os, subprocess, shutil
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import obonet
import pandas as pd
import torch
import pickle
from biopandas.pdb import PandasPdb
from collections import Counter
import csv
from sklearn.metrics import roc_curve, auc
# from torchviz import make_dot
from CONSTANTS import INVALID_ACIDS, amino_acids


def is_file(path):
    return os.path.exists(path)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def extract_id(header):
    return header.split('|')[1]


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


def remove_ungenerated_esm2_daisy_script(fasta_file, generated_directory):
    import os
    # those generated
    gen = os.listdir(generated_directory)
    gen = set([i.split(".")[0] for i in gen])

    seq_records = []

    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for record in input_seq_iterator:
        uniprot_id = extract_id(record.id)
        seq_records.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))

    print(len(seq_records), len(gen), len(set(seq_records).difference(gen)))

def filtered_sequences(fasta_file):
    """
         Script is used to create fasta files based on alphafold sequence, by replacing sequences that are different.
        :param fasta_file:
        :return: None
        """

    seq_records = []

    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    for record in input_seq_iterator:
        uniprot_id = extract_id(record.id)
        seq_records.append(create_seqrecord(id=uniprot_id, seq=str(record.seq)))

    SeqIO.write(seq_records, "data/Fasta/id2.fasta", "fasta")


def readlines_cluster(in_file):
    file = open(in_file)
    lines = [set(line.strip("\n").split("\t")) for line in file.readlines() if line.strip()]
    file.close()
    return lines


def read_dictionary(file):
    reader = csv.reader(open(file, 'r'), delimiter='\t')
    d = {}
    for row in reader:
        k, v = row[0], row[1]
        d[k] = v
    return d


def get_proteins_from_fasta(fasta_file):
    proteins = list(SeqIO.parse(fasta_file, "fasta"))
    proteins = [i.id for i in proteins]
    return proteins


def read_cafa5_scores(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    return lines


def fasta_to_dictionary(fasta_file, identifier='protein_id'):
    if identifier == 'protein_id':
        loc = 1
    elif identifier == 'protein_name':
        loc = 2
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        if "|" in seq_record.id:
            data[seq_record.id.split("|")[loc]] = (
            seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
        else:
            data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def get_sequence_from_pdb(pdb_file, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(pdb_file)

    pdb_df = pdb_to_pandas.df['ATOM']

    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & ((pdb_df['chain_id'])[0] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    residues = pdb_df['residue_name'].to_list()
    residues = ''.join([amino_acids[i] for i in residues if i != "UNK"])
    return residues


def is_ok(seq, MINLEN=49, MAXLEN=1022):
    """
           Checks if sequence is of good quality
           :param MAXLEN:
           :param MINLEN:
           :param seq:
           :return: None
           """
    if len(seq) < MINLEN or len(seq) >= MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def class_distribution_counter(**kwargs):
    """
        Count the number of proteins for each GO term in training set.
    """
    data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

    all_proteins = []
    for i in data:
        all_proteins.extend(data[i])

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    annot = annot.where(pd.notnull(annot), None)
    annot = annot[annot['Protein'].isin(all_proteins)]
    annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()

    terms = []
    for i in annot:
        terms.extend(annot[i].split(","))

    counter = Counter(terms)

    # for i in counter.most_common():
    #     print(i)
    # print("# of ontologies is {}".format(len(counter)))

    return counter


def save_ckp(state, is_best, checkpoint_dir):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = checkpoint_dir + "current_checkpoint.pt"
    best_model_path = checkpoint_dir + "best_model.pt"
    # save checkpoint data_bp to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


# def load_ckp_model_only(checkpoint_dir, model, best_model=False):
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     if best_model:
#         checkpoint_fpath = checkpoint_dir + "best_checkpoint.pt"
#     else:
#         checkpoint_fpath = checkpoint_dir + "current_checkpoint.pt"

#     if os.path.exists(checkpoint_fpath):
#         print("Loading model checkpoint @ {}".format(checkpoint_fpath))
#         checkpoint = torch.load(checkpoint_fpath)
#         model.load_state_dict(checkpoint['state_dict'])
#     return model


def load_ckp(checkpoint_dir, model, optimizer=None, lr_scheduler=None, best_model=False, model_only=False):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    if best_model:
        checkpoint_fpath = checkpoint_dir + "best_model.pt"
    else:
        checkpoint_fpath = checkpoint_dir + "current_checkpoint.pt"

    checkpoint = torch.load(checkpoint_fpath,  map_location="cpu")

    model.load_state_dict(checkpoint['state_dict'])

    # initialize optimizer from checkpoint to optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # initialize lr scheduler from checkpoint to optimizer
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss

    if model_only:
        return model
    
    return model, optimizer, lr_scheduler, checkpoint['epoch'], valid_loss_min


def draw_architecture(model, data_batch):
    '''
    Draw the network architecture.
    '''
    output = model(data_batch)
    make_dot(output, params=dict(model.named_parameters())).render("rnn_lstm_torchviz", format="png")


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_graph(obo_file):
    go_graph = obonet.read_obo(open(obo_file, 'r'))

    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)

    print("Number of nodes: {}, edges: {}".format(len(go_graph.nodes), len(go_graph.edges)))
    go_graph.remove_edges_from(unaccepted_edges)
    print("Number of nodes: {}, edges: {}".format(len(go_graph.nodes), len(go_graph.edges)))

    return go_graph