## Train Dataset
```
Downloaded from: http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_all.gaf.212.gz
```


## Test Dataset
```
Downloaded from: http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_all.gaf.218.gz
date-generated: 2023-12-04 09:42
```


#### Preprocessed Train/Validation Data Description
```
train_validation fasta: TFewData/ont/train_sequences.fasta
test fasta: test_fasta.fasta

order of ontologies: TFewData/ont/sorted_terms.pickle
Index of ontology in model: term_indicies.pickle
train proteins: TFewData/ont/train_proteins.pickle
validation proteins: TFewData/ont/validation_proteins.pickle
train and validation proteins: TFewData/ont/all_proteins.pickle
label data: TFewData/ont/graph.pt

processed dataset:
    TFewData/ont/train_data.pickle
    TFewData/ont/validation_data.pickle

    Format:
    Dictionary containing the preprocessed. We explored esm, msa & interpro. 
    Each index contains the data for same protein
        dictionary{
            labels: [], labels
            'esm2_t48': [], :-> esm data
            'msa_1b': [] :-> msa data
            'interpro': [], :-> Interpro data
            'diamond': [], :-> Did not use
            'string': [], :-> Did not use
            'protein: [] :-> protein name
        }

```


#### Preprocessed Test Data Description
```
Created with create_test.py script, adapted from
https://github.com/nguyenngochuy91/CAFA_benchmark(create_benchmark.py)

Inputs:
    t1: goa_uniprot_all.gaf.212.gz
    t2: goa_uniprot_all.gaf.218.gz


TFewData/test/t2/test_proteins:
(NK or LK) for each ontology (bp,cc,mf).

TFewData/test/t2/groundtruth:
Test groundtruth


Predictions from various models compared are kept in:
TFewData/evaluation
```


#### Trained models
```
TFewData/ont/full_gcn :-> final model
TFewData/ont/models/label/GCN :-> label embedding model

All other models will be uploaded to zenodo soon.
```