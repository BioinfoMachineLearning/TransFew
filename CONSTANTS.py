residues = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,
    "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20
}

INVALID_ACIDS = {"U", "O", "B", "Z", "J", "X", "*"}

amino_acids = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "PYL": "O", "SER": "S", "SEC": "U", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "ASX": "B", "GLX": "Z", "XAA": "X", "XLE": "J"
}

root_terms = {"GO:0008150", "GO:0003674", "GO:0005575"}

exp_evidence_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"}

# ROOT_DIR = "/home/fbqc9/Workspace/DATA/"
ROOT_DIR = "/home/fbqc9/Workspace/TFewData/"

ROOT = "/home/fbqc9/PycharmProjects/TransFun2/"

NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process"
}

FUNC_DICT = {
    'cc': 'GO:0005575',
    'mf': 'GO:0003674',
    'bp': 'GO:0008150'
}

BENCH_DICT = {
    'cc': "CCO",
    'mf': 'MFO',
    'bp': 'BPO'
}

NAMES = {
    "cc": "Cellular Component",
    "mf": "Molecular Function",
    "bp": "Biological Process"
}

GO_FILTERS = {
    'cc': (25, 4),
    'mf': (30, 4),
    'bp': (30, 8)
}

go_graph_path = ROOT_DIR + "/obo/go-basic.obo"