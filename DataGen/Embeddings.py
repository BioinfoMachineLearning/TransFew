# generate embedding from esm sequence
import subprocess
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


def generate_bulk_embedding(fasta):
    # name model output dir, embedding layer 1, embedding layer 2, batch
    models = (("esm_msa_1b", "esm_msa1b_t12_100M_UR50S", "msa", "esm_msa_1b", 11, 12, 10),
                ("esm_2", "esm2_t48_15B_UR50D", fasta, "esm2_t48", 47, 48, 100),
                ("esm_2", "esm2_t36_3B_UR50D", fasta, "esm_t36", 35, 36, 4096))
    model = models[1]
    CMD = "python -u {} {} {} /home/fbqc9/{} --repr_layers {} {} --include mean per_tok " \
                    "--toks_per_batch {}".format("external/extract.py", model[1], model[2], \
                                                model[3], model[4], model[5], model[6])

    print(CMD)
    subprocess.call(CMD, shell=True, cwd="./")


def generate_embeddings(fasta_path):
    def merge_pts(keys, fasta):
        embeddings = [47, 48]
        for pos, protein in enumerate(keys):
            print(pos, protein)
            fasta_dic = fasta_to_dictionary(fasta)

            tmp = []
            for level in range(keys[protein]):
                os_path = "/home/fbqc9/esm2_t48/{}_{}.pt".format(protein, level)
                tmp.append(torch.load(os_path))

            data = {'representations': {}, 'mean_representations': {}}
            for index in tmp:
                for rep in embeddings:
                    # print(index['mean_representations'][rep].shape, torch.mean(index['representations'][rep], dim=0).shape)
                    assert torch.equal(index['mean_representations'][rep], torch.mean(index['representations'][rep], dim=0))

                    if rep in data['representations']:
                        data['representations'][rep] = torch.cat((data['representations'][rep], index['representations'][rep]))
                    else:
                        data['representations'][rep] = index['representations'][rep]

            for emb in embeddings:
                assert len(fasta_dic[protein][3]) == data['representations'][emb].shape[0]

            for rep in embeddings:
                data['mean_representations'][rep] = torch.mean(data['representations'][rep], dim=0)

            # print("saving {}".format(protein))
            torch.save(data, "/home/fbqc9/esm2_t48/{}.pt".format(protein))

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

    keys = {}
    sequences = []
    input_seq_iterator = SeqIO.parse(fasta_path, "fasta")
    for record in input_seq_iterator:
        if len(record.seq) > 1021:
            _seqs, _keys = crop_fasta(record)
            sequences.extend(_seqs)
            keys.update(_keys)
        else:
            sequences.append(record)

    cropped_fasta = "temp.fasta"
    SeqIO.write(sequences, cropped_fasta, "fasta")

    # generate_bulk_embedding(cropped_fasta)

    # merge
    if len(keys) > 0:
        print("Found {} protein with length > 1021".format(len(keys)))
        merge_pts(keys, fasta_path)


# fasta_path = "/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta"
fasta_path = "/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta"
generate_embeddings(fasta_path)
