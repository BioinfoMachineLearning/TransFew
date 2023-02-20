import csv
import os
import subprocess

import numpy as np
import obonet
import pandas as pd
import torch
from Bio.Seq import Seq
from Bio import SeqIO, SwissProt
from Bio.SeqRecord import SeqRecord

import CONSTANTS
from preprocessing.utils import get_sequence_from_pdb, pickle_save, pickle_load, create_seqrecord, fasta_to_dictionary

exp_evidence_codes = set([
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
    "HTP", "HDA", "HMP", "HGI", "HEP"])


def read_uniprot(in_file, save=False, out_file="uniprot"):
    handle = open(in_file)
    all = [["ACC", "ID", "DESCRIPTION", "WITH_STRING", "EVIDENCE", "GO_ID", "ORGANISM", "TAXONOMY", "DATA CLASS",
            "CREATED",
            "SEQUENCE UPDATE", "ANNOTATION UPDATE", "SEQUENCE"]]
    for record in SwissProt.parse(handle):
        primary_accession = record.accessions[0]
        entry_name = record.entry_name
        cross_refs = record.cross_references
        organism = record.organism
        taxonomy = record.taxonomy_id
        assert len(taxonomy) == 1
        taxonomy = taxonomy[0]
        data_class = record.data_class
        created = record.created[0]
        sequence_update = record.sequence_update[0]
        annotation_update = record.annotation_update[0]
        sequence = record.sequence
        for ref in cross_refs:
            if ref[0] == "GO":
                assert len(ref) == 4
                go_id = ref[1]
                description = ref[2]
                evidence = ref[3].split(":")
                with_string = evidence[1]
                evidence = evidence[0]
                if evidence in exp_evidence_codes:
                    all.append(
                        [primary_accession, entry_name, description, with_string,
                         evidence, go_id, organism, taxonomy, data_class, created,
                         sequence_update, annotation_update, sequence])
    if save:
        with open(out_file, "w") as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerows(all)

    else:
        df = pd.DataFrame(all[1:], columns=all[0])
        return df


def get_files(pth):
    import os
    x = os.listdir(pth)
    return x


def add_ontology(dataframe, obo):
    ontologies = []
    go_graph = obonet.read_obo(obo, 'r')
    for line_number, (index, row) in enumerate(dataframe.iterrows()):
        go_id = row[5]
        namespace = go_graph.nodes[go_id]['namespace']
        ontologies.append(namespace)
    return ontologies


def sequence_length(row):
    return len(row[14])

def test_statistics(input_file):
    def add_domain(dataframe):
        domains = []
        domain_keys = {}
        for i in Constants.DOMAINS.items():
            for j in i[1]:
                domain_keys[j] = i[0]
        for line_number, (index, row) in enumerate(dataframe.iterrows()):
            taxonomy = row[8]
            domains.append(domain_keys.get(taxonomy, ""))
        print(domains)
        return domains

    data = pd.read_csv(input_file, sep="\t")

    print("Columns")
    print(data.columns)
    print("##############################\n\n")

    print("Ontology statistics")
    print(data.groupby(['ONTOLOGY'])['ACC'].nunique())
    print("##############################\n\n")

    orgs = [9606, 559292, 10090]
    print("Taxonomy statistics")
    print(data.loc[data['TAXONOMY'].isin(orgs)].groupby(['ORGANISM', 'ONTOLOGY'])['ACC'].nunique())
    print("##############################\n\n")

    print("Domain statistics")
    data['DOMAINS'] = np.array(add_domain(data))
    print(data.head())
    # print(data.loc[data['TAXONOMY'].isin(orgs)].groupby(['ORGANISM', 'ONTOLOGY'])['ACC'].nunique())
    print("##############################\n\n")

    # Domains Statistics


def query(proteins):
    for protein in proteins:
        output = Constants.ROOT + "downloaded/AF-{}-F1-model_v2.pdb".format(protein)
        url = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v2.pdb".format(protein)
        print(url)
        subprocess.call('wget -O {} {}'.format(output, url), shell=True)


def extract_sequences(proteins):
    cut_sequences = []
    sequences = []
    missing = []
    for protein in proteins:
        try:
            print(protein)
            seq = get_sequence_from_pdb(Constants.ROOT + "alphafold/AF-{}-F1-model_v2.pdb.gz".format(protein), "A")
            sequences.append(create_seqrecord(id="{}|{}".format(protein, protein), description=protein, seq=str(seq)))
            for j in range(int(len(seq) / 1021) + 1):
                tmp = protein + "_____" + str(j)
                id = "timebased|{}|{}".format(tmp, protein)
                cut_seq = str(seq[j * 1021:(j * 1021) + 1021])
                cut_sequences.append(create_seqrecord(id=id, description=protein, seq=str(cut_seq)))
        except FileNotFoundError:
            missing.append(protein)
    SeqIO.write(cut_sequences, Constants.ROOT + "/timebased/{}.fasta".format("cut"), "fasta")
    SeqIO.write(sequences, Constants.ROOT + "/timebased/{}.fasta".format("sequences"), "fasta")
    pickle_save(missing, Constants.ROOT + "timebased/missing_proteins")


def generate_bulk_embedding(fasta_file, output_dir, path_to_extract_file):
    subprocess.call('python extract.py esm1b_t33_650M_UR50S {} {} --repr_layers 0 32 33 '
                    '--include mean per_tok --truncate'.format("{}".format(fasta_file),
                                                               "{}".format(output_dir)),
                    shell=True, cwd="{}".format(path_to_extract_file))


def merge_pts(pdbs):
    unique_files = {i.split("_____")[0] for i in pdbs}
    levels = {0, 1, 2}
    embeddings = [0, 32, 33]

    pth = "/data/pycharm/TransFunData/data/timebased/esm_merged"
    created = get_files(pth)

    created = [i.split('.')[0] for i in created]
    print(len(created))

    for i in unique_files:
        if not i in created:
            fasta = fasta_to_dictionary(Constants.ROOT + "timebased/sequences.fasta")
            tmp = []
            for j in levels:
                os_path = Constants.ROOT + 'timebased/esm/{}_____{}.pt'.format(i, j)
                if os.path.isfile(os_path):
                    tmp.append(torch.load(os_path))

            data = {'representations': {}, 'mean_representations': {}}
            for index in tmp:
                splits = index['label'].split("|")
                data['label'] = splits[0] + "|" + splits[1].split("_____")[0] + "|" + splits[2]

                for rep in embeddings:
                    assert torch.equal(index['mean_representations'][rep],
                                       torch.mean(index['representations'][rep], dim=0))

                    if rep in data['representations']:
                        data['representations'][rep] = torch.cat(
                            (data['representations'][rep], index['representations'][rep]))
                    else:
                        data['representations'][rep] = index['representations'][rep]

            print(len(fasta[i][3]), data['representations'][33].shape[0])
            assert len(fasta[i][3]) == data['representations'][33].shape[0]

            for rep in embeddings:
                data['mean_representations'][rep] = torch.mean(data['representations'][rep], dim=0)

            print("saving {}".format(i))
            torch.save(data, Constants.ROOT + "timebased/esm_merged/{}.pt".format(i))


#
# x = set(x['ACC'].to_list())
# ct = []
# for i in x:
#     try:
#         seq = get_sequence_from_pdb(Constants.ROOT + "alphafold/AF-{}-F1-model_v2.pdb.gz".format(i), "A")
#     except FileNotFoundError:
#         ct.append(i)
# except IndexError:
#     os.remove(Constants.ROOT + "alphafold/AF-{}-F1-model_v2.pdb.gz".format(i))

# x = pickle_load("ct")
# print(len(x))
# query(x)

# print("Collecting Old")
# old_df = read_uniprot(Constants.ROOT + "uniprot/uniprot_sprot.dat", save=False, out_file="old_uniprot")
# old_proteins = set(old_df['ACC'].to_list())
# #
# print("Collecting New")
# new_df = read_uniprot(Constants.ROOT + "timebased/uniprot_sprot.dat", save=False, out_file="new_uniprot")
# new_df = new_df[~new_df['ACC'].isin(old_proteins)]


def tmp(in_file):
    handle = open(in_file)
    some_dict = {}
    for record in SwissProt.parse(handle):
        primary_accession = record.accessions[0]
        sequence = record.sequence
        some_dict[primary_accession] = sequence
    return some_dict

some_dic = tmp(Constants.ROOT + "uniprot/uniprot_sprot.dat")

data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")

data['SEQUENCE'] = data['ACC'].map(some_dic)
data['SEQUENCE LENGTH'] = data.apply(lambda row: sequence_length(row), axis=1)

data.to_csv(Constants.ROOT + "timebased/test_data_new", sep="\t")

print(data)
exit()
#
# print("Adding Ontology")
# new_df['ONTOLOGY'] = np.array(add_ontology(new_df, Constants.ROOT+"timebased/go-basic.obo"))

# print("SEQUENCE LENGTH")
# new_df['SEQUENCE LENGTH'] = new_df.apply (lambda row: sequence_length(row), axis=1)

# print(new_df[['ACC', 'TAXONOMY', 'ONTOLOGY']])
# print(new_df.shape)
#
# print("Saving")
# new_df.to_csv(Constants.ROOT + "timebased/test_data", sep="\t")


# test_statistics(Constants.ROOT + "timebased/test_data")

# data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")
# missing = set(pickle_load(Constants.ROOT + "timebased/missing_proteins"))
#
# data = set(data['ACC'].to_list()).difference(missing)
# extract_sequences(data)

# generate_bulk_embedding(Constants.ROOT + "/timebased/{}.fasta".format("cut"),
#                         Constants.ROOT + "/timebased/esm", "/data/pycharm/TransFun/preprocessing")

# merge_pts(data)

# x = set(x['ACC'].to_list())
# ct = []
# for i in x:
#     try:
#         seq = get_sequence_from_pdb(Constants.ROOT + "alphafold/AF-{}-F1-model_v2.pdb.gz".format(i), "A")
#     except FileNotFoundError:
#         ct.append(i)
# except IndexError:
#     os.remove(Constants.ROOT + "alphafold/AF-{}-F1-model_v2.pdb.gz".format(i))

# print(len(ct))
# pickle_save(ct, "ct")

# x = pickle_load("ct")
#
# for i in x:
#     print(i)