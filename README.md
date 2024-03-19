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
```
Predict protein functions with TransFew

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to data files (models)
  --working-dir WORKING_DIR
                        Path to generate temporary files
  --ontology ONTOLOGY   Path to data files
  --no-cuda NO_CUDA     Disables CUDA training.
  --batch-size BATCH_SIZE
                        Batch size.
  --fasta-path FASTA_PATH
                        Path to Fasta
  --output OUTPUT       File to save output
  
```

4. An example of predicting cellular component of some proteins: 
```
    python predict.py --fasta-path path-to-fasta-file --data-path path-to-data-directory --ontology cc/mf/bp  --output result.txt
```


## Training
1. To reproduce/train
```
    1. Change ROOT_DIR in CONSTANTS.py to path of data
    2. Run: python training.py
```



## Reference
```


```


