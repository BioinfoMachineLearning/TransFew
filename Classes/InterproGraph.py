import networkx as nx


class Interpro:
    '''
        Class to handle interpro data
    '''
    def __init__(self, file):
        self.lines = None
        self.file = file
        self.graph = nx.DiGraph()

    def propagate_graph(self):
        self.read_file()

        for node in self.lines:
            # 8 dashes
            if node[0].startswith("--------"):
                l5 = node[0].strip("--------")
                self.graph.add_edges_from([(l4, l5)])
            # 6 dashes
            elif node[0].startswith("------"):
                l4 = node[0].strip("------")
                self.graph.add_edges_from([(l3, l4)])
            # 4 dashes
            elif node[0].startswith("----"):
                l3 = node[0].strip("----")
                self.graph.add_edges_from([(l2, l3)])
            # 2 dashes
            elif node[0].startswith("--"):
                l2 = node[0].strip("--")
                self.graph.add_edges_from([(l1, l2)])
            else:
                l1 = node[0]
                if not self.graph.has_node(l1):
                    self.graph.add_node(l1)

    def read_file(self):
        rels = open(self.file, 'r')
        self.lines = [i.rstrip('\n').split("::") for i in rels.readlines()]

    def get_graph(self):
        return self.graph


_graph = Interpro("../data/ParentChildTreeFile.txt")
_graph.propagate_graph()
graph = _graph.get_graph()



