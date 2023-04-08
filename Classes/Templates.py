import numpy as np

import CONSTANTS
from Utils import is_file
from preprocessing.utils import pickle_load, pickle_save


class Templates:
    """
    This class is used to handle all template generations
    """

    def __init__(self, **kwargs):
        self.dir = kwargs.get('directory', CONSTANTS.ROOT_DIR + "datasets/")
        self.file_name = kwargs.get('file_name', "template")
        self.scaling = kwargs.get('scaling', "scaled")  # norm / scaling
        self.template = {}
        self.diamond = kwargs.get('diamond_path', CONSTANTS.ROOT_DIR + "diamond/output.tsv")

    def generate_templates(self):
        if not is_file(self.dir + "template"):
            self._generate_template()
        else:
            self.template = pickle_load(self.dir + "template")

    def _generate_template(self):
        ontologies = ("cc", "mf", "bp")
        tmp = pickle_load(CONSTANTS.ROOT_DIR + "datasets/training_validation")
        for ontology in ontologies:
            self.template[ontology] = {}
            proteins = tmp[ontology]['train'].union(tmp[ontology]['valid'])
            labels = pickle_load(CONSTANTS.ROOT_DIR + "datasets/labels")[ontology]

            # BLAST Similarity
            diamond_scores = {}
            with open(self.diamond) as f:
                for line in f:
                    it = line.strip().split()
                    if it[0] not in diamond_scores:
                        diamond_scores[it[0]] = {}
                    diamond_scores[it[0]][it[1]] = float(it[2])

            if self.scaling == 'norm':
                pass
            elif self.scaling == 'scaled':
                for protein in proteins:
                    if protein in diamond_scores:
                        sim_prots = diamond_scores[protein]
                        neighbours = sim_prots.items()

                        neighbour_score = []
                        go_scores = []
                        for neighbour, _score in neighbours:
                            if neighbour in labels and _score < 100:
                                go_scores.append(labels[neighbour])
                                neighbour_score.append(_score / 100)
                        go_scores = np.array(go_scores)
                        neighbour_score = np.array(neighbour_score)

                        _score = np.matmul(neighbour_score, go_scores)

                        if len(_score.shape) == 0:
                            _score = np.zeros(len(labels[protein]))

                    else:
                        _score = np.zeros(len(labels[protein]))

                    self.template[ontology][protein] = _score

        pickle_save(self.template, self.dir + "template")

    def get(self):
        return self.template
