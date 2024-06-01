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
    return dic, all_protein_name


def read_gpi(in_file, proteins):
    results = {'trembl': set(), 'swissprot':set()}
    with open(in_file, 'r') as handle:
        for entry in GOA.gpi_iterator(handle):
            if entry['DB'] == 'UniProtKB' and entry['DB_Object_ID'] in proteins:
                if entry['Gene_Product_Properties'][0] == "db_subset=TrEMBL":
                    results['trembl'].add(entry['DB_Object_ID'])
                elif entry['Gene_Product_Properties'][0] == "db_subset=Swiss-Prot":
                    results['swissprot'].add(entry['DB_Object_ID'])
    return results

    
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


def write_annotations(data_dic, sptr, pth):
    sprt_dic = {j: i for i in sptr for j in sptr[i]}
    '''
        data_dic: annotations in dictionary
        sptr: swissprot or trembl
    '''
    # Did not find sequence 
    to_remove = {'C0HM98', 'C0HM97', 'C0HMA1', 'C0HM44'}
    
    go_graph = get_graph(CONSTANTS.go_graph_path)

    ontologies = ['mf', 'cc', 'bp']

    ontology_map = {'F': 'mf', 'P': 'bp', 'C':'cc'}

    test_proteins = {i: {'trembl': set(), 'swissprot': set()} for i in ontologies}
    groundtruth = {i: {} for i in ontologies}

    for ont in data_dic:
        for acc, terms in data_dic[ont].items():
            if acc in to_remove:
                pass
            else: 
                for term in terms:
                    try:
                        tmp = nx.descendants(go_graph, term).union(set([term]))
                        if acc in groundtruth[ontology_map[ont]]:
                            groundtruth[ontology_map[ont]][acc].update(tmp)
                        else:
                            groundtruth[ontology_map[ont]][acc] = tmp
                        
                        test_proteins[ontology_map[ont]][sprt_dic[acc]].add(acc)
                    except nx.exception.NetworkXError:
                        pass

    for ont in ontologies:
        for ts, prots   in test_proteins[ont].items():
            file_name = pth + "/{}_{}.tsv".format(ont, ts)
            file_out = open(file_name, 'w')
            for prot in prots:
                for annot in groundtruth[ont][prot]:
                    file_out.write(prot + '\t' + annot + '\n')
            file_out.close()
    
    aspect_dict = {'bp': 'BPO', 'cc': 'CCO', 'mf': 'MFO'}

    file_out = open( pth + "/all_groundtruth.tsv", 'w')
    for ont in groundtruth:
        for protein in groundtruth[ont]:
            for term in groundtruth[ont][protein]:
                file_out.write(protein + '\t' + term  + '\t' + aspect_dict[ont] + '\n')

    file_out.close()

    

    # pickle_save(test_proteins, pth + "/test_proteins")
    # pickle_save(groundtruth, pth + "/groundtruth")



def generate(t1, t2):

    if not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_dic.pickle".format(t1, t1)) \
                or not is_file(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}.pickle".format(t1, t1)):
        t1_dic, all_protein_t1 = read_gaf(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gaf.212".format(t1))
        print("Reading GAF file 1")
        pickle_save(t1_dic, CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t1, t1))
        pickle_save(all_protein_t1, CONSTANTS.ROOT_DIR + "test/t1/all_protein_{}".format(t1))
    else:
        t1_dic = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t1, t1))
        all_protein_t1 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t1, t1))

    if not is_file(CONSTANTS.ROOT_DIR + "test/{}/{}_dic.pickle".format(t2, t2)) \
                or not is_file(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}.pickle".format(t2, t2)):
        print("Reading GAF file 2")
        t2_dic, all_protein_t2 = read_gaf(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gaf".format(t2))
        pickle_save(t2_dic, CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t2, t2))
        pickle_save(all_protein_t2, CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t2, t2))
    else:
        t2_dic = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/{}_dic".format(t2, t2))
        all_protein_t2 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/all_protein_{}".format(t2, t2))


    # Find trembl and swissprt proteins
    sptr_pth = CONSTANTS.ROOT_DIR + "test/{}/sptr".format(t1)
    if not is_file(sptr_pth+".pickle"):
        print("Reading GPI file 1")
        sptr1 = read_gpi(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gpi.212".format(t1), all_protein_t1)
        pickle_save(sptr1, sptr_pth)
    else:
        sptr1 = pickle_load(sptr_pth)


    sptr_pth = CONSTANTS.ROOT_DIR + "test/{}/sptr".format(t2)
    if not is_file(sptr_pth+".pickle"):
        print("Reading GPI file 2")
        sptr2 = read_gpi(CONSTANTS.ROOT_DIR + "test/{}/goa_uniprot_all.gpi.218".format(t2), all_protein_t2)
        pickle_save(sptr2, sptr_pth)
    else:
        sptr2 = pickle_load(sptr_pth)


    NK_dic, LK_dic = analyze(t1_dic, t2_dic, all_protein_t1)

    out_pth = CONSTANTS.ROOT_DIR + "test/output_{}_{}".format(t1, t2)
    create_directory(out_pth)

    NK_LK_dic = {}
    NK_LK_dic['P'] = NK_dic['P'] | LK_dic['P']
    NK_LK_dic['C'] = NK_dic['C'] | LK_dic['C']
    NK_LK_dic['F'] = NK_dic['F'] | LK_dic['F']


    write_annotations(NK_LK_dic, sptr2, out_pth)



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
        



##### Generate for test dataset #########
# generate no known & limited known
generate(t1="t1", t2="t2")




exit()

all_test = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format(prefix))
all_test = set([j for i in all_test for j in all_test[i]])


all_test1 = pickle_load(CONSTANTS.ROOT_DIR + "test/{}/test_proteins".format("t2"))
all_test1 = set([j for i in all_test1 for j in all_test1[i]])


print(len(all_test.difference(all_test1)), len(all_test.intersection(all_test1)))
get_fasta(all_test.difference(all_test1))



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


# create_test_dataset()