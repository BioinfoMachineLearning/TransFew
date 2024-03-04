import os
import numpy as np
import pandas as pd
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import CONSTANTS
from Classes.Diamond import Diamond
from Classes.Fasta import Fasta
from Classes.Interpro import Interpro, create_indicies
from Classes.STRING import STRING
from Dataset.Dataset import TransFunDataset
from Utils import count_proteins, get_proteins_from_fasta, pickle_load, pickle_save





'''infiles = ["test_proteins_1800", "test_proteins_3600",  "test_proteins_5400",
            "test_proteins_900", "test_proteins_2700",  "test_proteins_4500",
            "test_proteins_5641",  "test_proteins.out"]

# merge chunks
df = pd.DataFrame()
prefix = "/home/fbqc9/testinetrpro/{}"
for infile in infiles:
    print(infile)
    data = pd.read_csv(prefix.format(infile), sep="\t",
                    names=["Protein accession", "Sequence MD5", "Sequence length", "Analysis",
                            "Signature accession", "Signature description", "Start location",
                            "Stop location", "Score", "Status", "Date",
                            "InterPro annotations", "InterPro annotations description ", "GO annotations"])
    print(data[['Protein accession', 'InterPro annotations']].head(3))
    df = pd.concat([df, data], axis=0)
# passed additional quality checks and is very unlikely to be a false match.
df = df[['Protein accession', 'InterPro annotations']]
df = df[df["InterPro annotations"] != "-"]

print(df[['Protein accession', 'InterPro annotations']].head(3))

df.to_csv(prefix.format("test_proteins.out"), index=False, sep="\t")


exit()'''


'''p1 = "/home/fbqc9/Workspace/DATA/interpro/test_proteins.out"
p2 = "/home/fbqc9/testinetrpro/test_proteins.out"
data = pd.read_csv(p1, sep="\t",
                            names=["Protein accession", "Sequence MD5", "Sequence length", "Analysis",
                                    "Signature accession", "Signature description", "Start location",
                                    "Stop location", "Score", "Status", "Date",
                                    "InterPro annotations", "InterPro annotations description ", "GO annotations"])

print(len(set(data["Protein accession"].to_list())))
exit()'''


'''all_test_proteins = set()
dta = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
for i in dta:
    all_test_proteins.update(dta[i])
all_test_proteins = list(all_test_proteins)
print(len(all_test_proteins))


for i in all_test_proteins:
    try:
        x = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))
        print(i, torch.sum(x['interpro_mf'].x))
        break
    except FileNotFoundError:
        pass

    

interpro = Interpro(ont='mf')

mf_interpro_data, mf_interpro_sig, _ = interpro.get_interpro_test()
ct = 0
for i in mf_interpro_data:
    print(sum(mf_interpro_data[i]))
    for j,k in zip(mf_interpro_sig, mf_interpro_data[i]):
        if k == 1:
            print(i, j , k)
            ct = ct + 1
    if ct == 5:
        exit()


exit()'''


'''to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
all_test_proteins = set()
dta = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
for i in dta:
    all_test_proteins.update(dta[i])
all_test_proteins = list(all_test_proteins.difference(to_remove))
print(len(all_test_proteins))

kwargs = {
    'split': 'selected',
    'proteins': all_test_proteins
}
train_dataset = TransFunDataset(**kwargs)

exit()




x = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format("A0A7I2V2R9"))

print(x)


x = torch.load("/bmlfast/frimpong/shared_function_data/esm_msa1b/{}.pt".format("Q75WF1"))
print(x)

exit()


x = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format("Q75WF1"))

print(x)

exit()'''

to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
all_test_proteins = set()
dta = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")
for i in dta:
    all_test_proteins.update(dta[i])
dt = list(all_test_proteins.difference(to_remove))
print(len(dt))


for i in dt:
    tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))
    mas = tmp['esm_msa1b'].x

    if len(mas.shape) == 3: 
        tmp['esm_msa1b'].x = torch.mean(mas, dim=1)
        print(mas.shape, tmp['esm_msa1b'].x.shape)
        torch.save(tmp, CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))


exit()

onts = ['cc', 'bp', 'mf']

for ont in onts:
    store = {'labels': [],
                 'esm2_t48': [],
                 'msa_1b': [],
                 'interpro': [],
                 'diamond': [],
                 'string': [],
                 'protein': []
                 }
    
    for po, i in enumerate(dt):
        print("{}, {}, {}".format(ont, i, po))
        tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(i))
        esm = tmp['esm2_t48'].x
        msa = torch.mean(tmp['esm_msa1b'].x, dim=0).unsqueeze(0).cpu()
        diamond = tmp['diamond_{}'.format(ont)].x
        diamond = torch.mean(diamond, dim=0).unsqueeze(0)
        interpro = tmp['interpro_{}'.format(ont)].x
        string_data = tmp['string_{}'.format(ont)].x
        string_data = torch.mean(string_data, dim=0).unsqueeze(0)

        assert esm.shape == torch.Size([1, 5120])
        assert msa.shape == torch.Size([1, 768])
        
        store['esm2_t48'].append(esm)
        store['msa_1b'].append(msa)
        store['diamond'].append(diamond)
        store['interpro'].append(interpro)
        store['string'].append(string_data)
        store['protein'].append(i)


    pickle_save(store, "com_data/{}.data_test".format(ont))



exit()
onts = ['cc', 'bp', 'mf']

for ont in onts:

    data = pickle_load("com_data/{}.data_test".format(ont))


    msa_data = data['msa_1b']

    for i in msa_data:
        if i.device != torch.device("cpu"):
            print(i.device)





'''
def generate_test_data():
    all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/test_proteins")

    lk = set()
    for i in all_test:
        if i.startswith('LK_'):
            for j in all_test[i]:
                lk.add(j)

    all_test = set([j for i in all_test for j in all_test[i]])

    

    


    for i in all_test:

        x = torch.load("/home/fbqc9/esm_msa1b/{}.pt".format(i))
        print(x['representations_12'].shape)




    exit()

    # check if all data is available

    # esm
    esm = os.listdir("/bmlfast/frimpong/shared_function_data/esm2_t48/")
    esm = set([i.split(".")[0] for i in esm])


    msa = os.listdir("/home/fbqc9/esm_msa1b/")
    msa = set([i.split(".")[0] for i in msa])

    a3ms = os.listdir("/bmlfast/frimpong/shared_function_data/a3ms/")
    a3ms = set([i.split(".")[0] for i in a3ms])


    print(len(all_test.difference(esm)), len(all_test.difference(msa)), \
          len(all_test.difference(a3ms)))

    exit()



generate_test_data()

exit()
'''