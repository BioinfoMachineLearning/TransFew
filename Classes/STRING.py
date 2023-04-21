class STRING:
    '''
        Class to handle STRING data
    '''

    def __init__(self, dbase, mapping):
        self.string_dbase = dbase
        self.mapping = mapping

    def extract_uniprot(self):
        pass


String = STRING("../data/interpro/ParentChildTreeFile.txt")
