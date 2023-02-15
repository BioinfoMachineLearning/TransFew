from Classes.InterproGraph import Interpro


def read_file(file):
    data = open(file, 'r')
    data = data.readlines()
    data = [i.split("\t")[0] for i in data]
    return data


data = set(read_file("data/entry.list"))

graph_data = Interpro("data/ParentChildTreeFile.txt")
graph_data.propagate_graph()
_nodes = set(graph_data.get_graph().nodes)

print(data.difference(_nodes))
