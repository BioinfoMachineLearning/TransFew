import pandas as pd


def stats_on_clusters(in_file):
    """
    Find the min and max proteins counts in cluster
    """
    file = open(in_file)
    lines = [line.strip("\n").split("\t") for line in file.readlines() if line.strip()]
    file.close()

    print(len(min(lines, key=len)), len(max(lines, key=len)))



# stats_on_clusters("D:/Workspace/python-3/TFUN/data/Embeddings/cluster/final_clusters.csv")