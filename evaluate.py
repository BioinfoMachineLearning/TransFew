import os
import subprocess
import CONSTANTS


# Evaluate all
def evaluate_group(obo_file, ia_file, prediction_folder, groundtruth, outdir):
    # command = "cafaeval go-basic.obo prediction_dir test_terms.tsv -ia IA.txt -prop fill -norm cafa -th_step 0.001 -max_terms 500"
    command = "cafaeval {} {} {} -ia {} -norm cafa -th_step 0.01  -out_dir {}".\
        format(obo_file, prediction_folder, groundtruth, ia_file, outdir)
    subprocess.call(command, shell=True)


obo_file = CONSTANTS.ROOT_DIR + "obo/go-basic.obo"
ia_file = CONSTANTS.ROOT_DIR + "test/output_t1_t2/ia_prop.txt"



db_subset = ["swissprot", "trembl"]
ontologies = ["cc", "mf", "bp"]


# full evaluation
def full_evaluation():
    prediction_folder = CONSTANTS.ROOT_DIR + "evaluation/predictions/full/{}_{}"
    groundtruth = CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruths/full/{}_{}.tsv"
    out_dir = "results/full/{}_{}"
    for ont in ontologies:
        for sptr in db_subset:
            print("Evaluating {} {}".format(ont, sptr))
            evaluate_group(obo_file=obo_file,
                        ia_file=ia_file,
                        prediction_folder=prediction_folder.format(sptr, ont),
                        groundtruth=groundtruth.format(ont, sptr),
                        outdir=out_dir.format(ont, sptr))
        

# 30% SeqID evaluation
def seq_30_evaluation():
    prediction_folder = CONSTANTS.ROOT_DIR + "evaluation/predictions/seq_ID_30/{}_{}"
    groundtruth = CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruths/seq_ID_30/{}_{}.tsv"
    out_dir = "results/seq_ID_30/{}_{}"
    for ont in ontologies:
        for sptr in db_subset:
            print("Evaluating {} {}".format(ont, sptr))
            evaluate_group(obo_file=obo_file,
                        ia_file=ia_file,
                        prediction_folder=prediction_folder.format(sptr, ont),
                        groundtruth=groundtruth.format(ont, sptr),
                        outdir=out_dir.format(ont, sptr))
            
    
# 30% SeqID evaluation
def components_evaluation():
    prediction_folder = CONSTANTS.ROOT_DIR + "evaluation/predictions/components/{}_{}"
    groundtruth = CONSTANTS.ROOT_DIR + "test/output_t1_t2/groundtruths/full/{}_{}.tsv"
    out_dir = "results/components/{}_{}"
    for ont in ontologies:
        for sptr in db_subset:
            print("Evaluating {} {}".format(ont, sptr))
            evaluate_group(obo_file=obo_file,
                        ia_file=ia_file,
                        prediction_folder=prediction_folder.format(sptr, ont),
                        groundtruth=groundtruth.format(ont, sptr),
                        outdir=out_dir.format(ont, sptr))

            

# full_evaluation()
# seq_30_evaluation()
components_evaluation()