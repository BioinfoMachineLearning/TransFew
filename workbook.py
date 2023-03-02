# Notebook to do surplus work
import os
import shutil

import CONSTANTS


def interpro_go_terms():
    with open(CONSTANTS.ROOT_DIR + "interpro/interpro2go") as file:
        next(file)
        next(file)
        next(file)
        next(file)
        next(file)
        terms = {}
        for line in file:
            go = line.strip().split(";")[1]
            ipr = line.strip().split(",")[0].split(" ")[0].split(":")[1]

            if go in terms:
                print(go)
                terms[ipr].append(go)
            else:
                terms[ipr] = [go]

    print(terms)


interpro_go_terms()
