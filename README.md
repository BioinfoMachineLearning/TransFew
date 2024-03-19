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
conda env create -f transfew.yaml
conda activate transfew
```

## Prediction
```
Predict protein functions with TransFew

options:
  -h, --help            show this help message and exit

  --data-path DATA_PATH  Path to data files (models)

  --working-dir WORKING_DIR  Path to generate temporary 
  files

  --ontology ONTOLOGY   Path to data files

  --no-cuda NO_CUDA     Disables CUDA training.

  --batch-size BATCH_SIZE Batch size.

  --fasta-path FASTA_PATH Path to Fasta

  --output OUTPUT       File to save output
```

4. An example of predicting cellular component of some proteins: 
```
  python predict.py  --data-path /TFewData/ --fasta-path output_dir/test_fasta.fasta --ontology cc --working-dir output_dir --output result.tsv
```

##### Output format
```
  protein   GO term  score
  A0A7I2V2M2	GO:0043227	0.996
  A0A7I2V2M2	GO:0043226	0.996
  A0A7I2V2M2	GO:0043229	0.996
  A0A7I2V2M2	GO:0043231	0.995

```

## Dataset
```
See DATASET.md for description of data
```



## Training
The training program is available in training.py, to train the model:
```
    1. Change ROOT_DIR in CONSTANTS.py to path of data directory
    2. Run: python training.py
```



## Reference
```


```


