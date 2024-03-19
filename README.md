# Github Repository for TransFew:  
#### Improving protein function prediction by learning and integrating representations of protein sequences and function labels

TransFew leaverages representations of both protein sequences and
function labels (Gene Ontology (GO) terms) to predict the function of proteins. 



## Installation
```
# clone project
git clone https://github.com/BioinfoMachineLearning/TransFew.git
cd TransFew/

# download trained models and test sample


# create conda environment
conda env create -f environment.yml
conda activate transfew
```


## Prediction
1. To predict protein function with protein structures in the PDB format as input (note: protein sequences are automatically extracted from the PDB files in the input pdb path).
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type pdb --pdb-path data/alphafold --output output_file --cut-off probability_threshold
```

2. To predict protein function with protein sequences in the fasta format and protein structures in the PDB format as input: 
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type fasta --pdb-path data/alphafold --fasta-path path_to_a_fasta_file --output result.txt --cut-off probability_threshold
```

3. Full prediction command: 
```
Predict protein functions with TransFun

optional arguments:
  -h, --help            Help message
  --data-path DATA_PATH
                        Path to store intermediate data files
  --ontology ONTOLOGY   GO function category: cellular_component, molecular_function, biological_process
  --no-cuda NO_CUDA     Disables CUDA training
  --batch-size BATCH_SIZE
                        Batch size
  --input-type {fasta,pdb}
                        Input data type: fasta file or PDB files
  --fasta-path FASTA_PATH
                        Path to a fasta containing one or more protein sequences
  --pdb-path PDB_PATH   Path to the directory of one or more protein structure files in the PDB format
  --cut-off CUT_OFF     Cut-off probability threshold to report function
  --output OUTPUT       A file to save output. All the predictions are stored in this file
  
```

4. An example of predicting cellular component of some proteins: 
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

5. An example of predicting molecular function of some proteins: 
```
    python predict.py --data-path data --ontology molecular_function --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

## Training
1. To reproduce/train
```
    1. Change ROOT_DIR in CONSTANTS.py to path of data
    2. Run: python training.py
```


## Dataset
1. To reproduce/train
```
    1. Change ROOT_DIR in CONSTANTS.py to path of data
    2. Run: python training.py
```



## Reference
```
@article{10.1093/bioinformatics/btad208,
    author = {Boadu, Frimpong and Cao, Hongyuan and Cheng, Jianlin},
    title = "{Combining protein sequences and structures with transformers and equivariant graph neural networks to predict protein function}",
    journal = {Bioinformatics},
    volume = {39},
    number = {Supplement_1},
    pages = {i318-i325},
    year = {2023},
    month = {06},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btad208},
    url = {https://doi.org/10.1093/bioinformatics/btad208},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/39/Supplement\_1/i318/50741489/btad208.pdf},
}

```


