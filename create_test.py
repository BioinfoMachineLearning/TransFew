import os
from Bio.UniProt import GOA
import torch
import CONSTANTS
import networkx as nx
from Utils import create_directory, get_graph, is_file, \
    pickle_load, pickle_save, count_proteins, get_proteins_from_fasta
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

'''
function : given a file handle, parse in using gaf format and return a dictionary
           that identify those protein with experimental evidence and the ontology
input    : file text
output   : dic (key: name of file (number), value is a big dictionary store info about the protein)
'''


def read_gaf(handle):
    name = handle.split(".")[-1]
    dic = {}
    all_protein_name = set()
    # evidence from experimental
    Evidence = {'Evidence': set(["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", 
                                 "HTP", "HDA", "HMP", "HGI", "HEP"])}

    with open(handle, 'r') as handle:
        for rec in GOA.gafiterator(handle):
            if GOA.record_has(rec, Evidence) and rec['DB'] == 'UniProtKB':
                all_protein_name.add(rec['DB_Object_ID'])
                if rec['DB_Object_ID'] not in dic:
                    dic[rec['DB_Object_ID']] = {rec['Aspect']: set([rec['GO_ID']])}
                else:
                    if rec['Aspect'] not in dic[rec['DB_Object_ID']]:
                        dic[rec['DB_Object_ID']][rec['Aspect']] = set([rec['GO_ID']])
                    else:
                        dic[rec['DB_Object_ID']][rec['Aspect']].add(rec['GO_ID'])
    return name, dic, all_protein_name


'''
function : given t1 dic, t2 dic, we provide the dic for NK, and LK dic for each ontology
input    : 2 dics
output   : NK,LK dictionary
'''
def analyze(t1_dic, t2_dic, all_protein_t1):
    NK_dic = {'P': {}, 'C': {}, 'F': {}}
    LK_dic = {'P': {}, 'C': {}, 'F': {}}
    # dealing with NK and LK

    for protein in t2_dic:
        ## check the protein in t2_dic but not appear in t1
        if protein not in t1_dic: # and protein in all_protein_t1:  ## this going to be in NK
            ### check which ontology got new annotated
            for ontology in t2_dic[protein]:
                NK_dic[ontology][protein] = t2_dic[protein][ontology]
        ## check the protein that in t2_dic and appear in t1
        elif protein in t1_dic:
            ## check if in t1, this protein does not have all 3 ontology
            ### if yes, then not include since full knowledge
            ### else
            if len(t1_dic[protein]) < 3:
                #### check if t2_dic include in the ontology that t1 lack of
                for ontology in t2_dic[protein]:
                    if ontology not in t1_dic[protein]:  # for those lack, include in LK
                        LK_dic[ontology][protein] = t2_dic[protein][ontology]
    return NK_dic, LK_dic


'''
function : given NK,LK dic , write out 6 files 
input    : 2 dics
output   : NK,LK dictionary
'''


def write_file(dic, knowledge, prefix="t2"):
    for ontology in dic:
        if ontology == 'F':
            name = CONSTANTS.ROOT_DIR + 'test/{}/output/'.format(prefix) + knowledge + '_mfo'
        elif ontology == 'P':
            name = CONSTANTS.ROOT_DIR + 'test/{}/output/'.format(prefix) + knowledge + '_bpo'
        elif ontology == 'C':
            name = CONSTANTS.ROOT_DIR + 'test/{}/output/'.format(prefix) + knowledge + '_cco'
        file_out = open(name, 'w')
        for protein in sorted(dic[ontology]):
            for annotation in dic[ontology][protein]:
                file_out.write(protein + '\t' + annotation + '\n')
        file_out.close()

        name = CONSTANTS.ROOT_DIR + 'test/{}/output/'.format(prefix) + knowledge
        file_out = open(name, 'w')
        for ontology in dic:
            for protein in sorted(dic[ontology]):
                for annotation in dic[ontology][protein]:
                    file_out.write(protein + '\t' + annotation + '\n')
        file_out.close()
    return None



def generate(t1, t2):
    if not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_name.pickle".format(t1, t1)) \
            or not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_dic.pickle".format(t1, t1)) \
                or not is_file(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}.pickle".format(t1, t1)):
        t1_name, t1_dic, all_protein_t1 = read_gaf(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gaf.212".format(t1))
        pickle_save(t1_name, CONSTANTS.ROOT_DIR + "test/{}/{}_name".format(t1, t1))
        pickle_save(t1_dic, CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t1, t1))
        pickle_save(all_protein_t1, CONSTANTS.ROOT_DIR + "test/t1/all_protein_{}".format(t1))
    else:
        t1_name = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_name".format(t1, t1))
        t1_dic = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t1, t1))
        all_protein_t1 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t1, t1))

    if not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_name.pickle".format(t2, t2)) \
            or not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_dic.pickle".format(t2, t2)) \
                or not is_file(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}.pickle".format(t2, t2)):
        print("Reading GAF file 2")
        t2_name, t2_dic, all_protein_t2 = read_gaf(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gaf".format(t2))
        pickle_save(t2_name, CONSTANTS.ROOT_DIR + "test/{}/{}_name".format(t2, t2))
        pickle_save(t2_dic, CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t2, t2))
        pickle_save(all_protein_t2, CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t2, t2))
    else:
        t2_name = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_name".format(t2, t2))
        t2_dic = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t2, t2))
        all_protein_t2 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t2, t2))

    NK_dic, LK_dic = analyze(t1_dic, t2_dic, all_protein_t1)

    create_directory(CONSTANTS.ROOT_DIR + "test/{}/output".format(t2))


    write_file(NK_dic, 'NK', t2)
    write_file(LK_dic, 'LK', t2)



def get_all_test_ground_truth(prefix="t2"):

    # objects
    go_graph = get_graph(CONSTANTS.go_graph_path)
    unfound = set()

    # test files
    chunks = ['LK_bpo', 'NK_bpo', 'LK_mfo', 'NK_mfo', 'LK_cco', 'NK_cco', 'LK', 'NK']

    test_proteins = {i: set() for i in chunks[:6]}
    groundtruth = {}
    for ch in chunks[:6]:
        infile = CONSTANTS.ROOT_DIR + "test/{}/output/{}".format(prefix, ch)
        file = open(infile)
        for line in file.readlines():
            acc, term = line.strip().split("\t")
            try:
                tmp = nx.descendants(go_graph, term).union(set([term]))

                if acc in groundtruth:
                    groundtruth[acc].update(tmp)
                else:
                    groundtruth[acc] = tmp
                
                test_proteins[ch].add(acc)
            except nx.exception.NetworkXError:
                unfound.add(term)

    pickle_save(test_proteins, CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format(prefix))
    pickle_save(groundtruth, CONSTANTS.ROOT_DIR + "test/{}/groundtruth".format(prefix))
    print(len(unfound))


def create_test_dataset():
    to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
    def get_ontology(test_set):
        if test_set == 'LK_bp' or test_set == 'NK_bp':
            return "bp"
        elif test_set == 'LK_mf' or test_set == 'NK_mf':
            return "mf"
        elif test_set == 'LK_cc' or test_set == 'NK_cc':
            return "cc"

    # load all test
    all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/t3/test_proteins")

    for test_set in all_test:

        store = {'labels': [],
                    'esm2_t48': [],
                    'msa_1b': [],
                    'interpro': [],
                    'diamond': [],
                    'string': [],
                    'protein': []
                    }
        ont = get_ontology(test_set)
        for pos, protein in enumerate(all_test[test_set].difference(to_remove)):
            print("{}, {}".format(pos, protein))
            tmp = torch.load(CONSTANTS.ROOT_DIR + "data/processed/{}.pt".format(protein))

            esm = tmp['esm2_t48'].x.squeeze(0).cpu()
            msa = torch.mean(tmp['esm_msa1b'].x, dim=0).cpu()#.unsqueeze(0).cpu()
            #diamond = tmp['diamond_{}'.format(ont)].x
            #diamond = torch.mean(diamond, dim=0)#.unsqueeze(0)
            interpro = tmp['interpro_{}'.format(ont)].x.squeeze(0)
            # string_data = tmp['string_{}'.format(ont)].x
            # string_data = torch.mean(string_data, dim=0).unsqueeze(0)

            assert esm.shape == torch.Size([5120])
            assert msa.shape == torch.Size([768])
            
            store['esm2_t48'].append(esm)
            store['msa_1b'].append(msa)
            store['diamond'].append(torch.zeros(5))
            store['interpro'].append(interpro)
            # store['string'].append(string_data)
            store['protein'].append(protein)

        pickle_save(store, CONSTANTS.ROOT_DIR + "test/t3/dataset/{}".format(test_set))



def get_fasta(proteins):
    input_seq_iterator = SeqIO.parse("/home/fbqc9/Workspace/DATA/uniprot/test_fasta.fasta", "fasta")
    save = []
    for pos, record in enumerate(input_seq_iterator):
        id = record.id
        if  id in proteins:
            save.append(SeqRecord(id=id, seq=record.seq, description=""))

    SeqIO.write(save, "/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta", "fasta")


def generate_single_fastas():
    input_seq_iterator = SeqIO.parse("/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta", "fasta")
    for pos, record in enumerate(input_seq_iterator):
        SeqIO.write(record, "/bmlfast/frimpong/shared_function_data/single_fastas_2/{}.fasta".format(record.id), "fasta")
        

'''
print(count_proteins("/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta"))
print(count_proteins("/home/fbqc9/Workspace/DATA/uniprot/test_fasta.fasta"))
exit()
'''

'''gen = os.listdir("/bmlfast/frimpong/shared_function_data/a3ms/")
gen = set([i.split(".")[0] for i in gen])

gen1 = os.listdir("/bmlfast/frimpong/shared_function_data/single_fastas2/")
gen1 = set([i.split(".")[0] for i in gen1])



diff = len(gen1.difference(gen))
print(diff)


exit()'''
# generate_single_fastas()
# exit()

##### Generate for t3 #########
# prefix="t3"
# generate no known & limited known
# generate(t1="t1", t2=prefix)


# generate test groundtruth
# get_all_test_ground_truth(prefix)

'''all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format(prefix))
all_test = set([j for i in all_test for j in all_test[i]])


all_test1 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format("t2"))
all_test1 = set([j for i in all_test1 for j in all_test1[i]])


print(len(all_test.difference(all_test1)), len(all_test.intersection(all_test1)))
get_fasta(all_test.difference(all_test1))'''



# get_fasta(all_test)

#x = count_proteins("/home/fbqc9/Workspace/DATA/uniprot/test_fasta_rem.fasta")
#print(x)

#exit()

# sequence_not_found({'C0HM44', 'C0HM98', 'C0HM97', 'C0HMA1'})




#exit()


'''all_test2 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format("t2"))


all_test = set([j for i in all_test for j in all_test[i]])

all_test2 = set([j for i in all_test2 for j in all_test2[i]])






gen = os.listdir("/bmlfast/frimpong/shared_function_data/esm2_t48/")
gen = set([i.split(".")[0] for i in gen])


diff = all_test - gen 

for i in diff:
    print(i)


exit()'''



# print(len(all_test))

"""
all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/test_proteins")
all_test = set([j for i in all_test for j in all_test[i]])
print(len(all_test))

all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/groundtruth")
print(len(all_test))
"""


create_test_dataset()