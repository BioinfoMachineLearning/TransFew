# Transfew Container for Protein Function Prediction

This Docker container implements a esm2 based Transfew protein function prediction pipeline. `Transfew` ileaverages representations of both protein sequences and function labels (Gene Ontology (GO) terms) to predict the function of proteins. It improves the accuracy of predicting both common and rare function terms (GO terms).

## &#128458; Contents
- `transfew_main.py` - Main orchestration script
- `run_transfew.sh`  - Shell scriot for antomatically active environment and excuate main orchestration script **transfew_main.py**
- `download_model.sh` - Shell script for automatically downloading model weights
- `MyDataset.py` - Customized dataloader
- `extract.py` - Extract embeddings from esm2
- `config.py` - Transfew Model Configuration
- `CONSTANTS.py` - Constant varibales Configuration
- `net_utils.py` - Auxiliary functions for Transfew Model
- `model.py` - Transfew main model
- `utils.py` - Auxiliary Functions for Data Processing
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition

## &#9881;&#65039; How It Works
**Step 1**: using `download_model.sh` download all the needed weights for diferent models to avoid re-downloading models.

**Step 2**: using `transfew_main.py` to complete the prediction, it orchestrates three stages to complete protein function prediction. 


- 1. **Transfew Embeddings**: The container first extract feature embeddings from the pretrained models and saved in a customized folder provided by user!
- 2. **Transfew Predictions**: Predict CC, MF, BP three kind terms with Transfew!
- 3. **Save Predictions**: Save the predicted results into a customized folder provided by user.


## &#128218; Input Dataset Format 

- &#129516; Sequence Format

  Sequences should be provided in **FASTA** format, an example can be found at current folder: 
**Example:** [`sequence.fasta`](./sequence.fasta)


## &#128640; Model caching

The `download_model.sh` script is used to download and cache all required model weights and metadata before running the container, ensuring they are available for prediction.

```bash
# The pretrained_transfew_model will be automatically downloaded to $PWD/checkpoints if no --cache-path is provided.
bash ./download_model.sh --cache-path /path/cache_folder
```

## &#128293; Building the Container
The container should be built after all scripts have been generated/collected. 


- From the funbind_container directory, the build process will take approximately one hour to complete:
`docker build --network=host -t transfew_predictor .`


- Alternatively, pull from [Docker Hub](https://hub.docker.com/r/yw7bh/transfew_predictor):
`docker pull yw7bh/transfew_predictor:latest`

## &#128293; Running the Container with Model Caching

```bash
docker run --rm \
  --name transfew_predictor \
  -v /path/to/test_data:/root/data \
  -v /path/to/test_output:/root/output \
  -v /path/to/checkpoints_at_local_server:/root/.cache/torch/hub/checkpoints \
  transfew_predictor \
  --fasta-path /root/data/sequence.fasta \
  --working-dir /root/output \
  --output prediction.tsv.gz \
```
**Notes:**
- Any model weights you download on the local server must be mounted to the Docker container at `/root/.cache/torch/hub/checkpoints`, otherwise, prediction will fail.

####  Required Arguments
- `--working-dir`: Path where all predicted results will be stored. This directory should be mounted to the corresponding output path on your local server. 
- `--output`: Name the prediction output file.
- `--fasta-path`: Path to the input file(sequence), a FASTA file containing query sequences to predict.

#### Optional Arguments


## &#128187; GPU Requirements

- **Recommended**: A100/H100
- **Minimum**: TBD
- **CPU fallback**: Will run on CPU but significantly slower

## &#128187; Model Information

- **Default Model**: `/root/.cache/torch/hub/checkpoints`
- **Model Size**: rootroximately 26 GB for the various Transfew pretrained models and ~29 GB for esm2_t48_15B_UR50D model.

- **First build and rebuild**: All required model weights and metadata will be downloaded and cached up front, reducing both the initial build time and any future rebuild times.

## &#128187; Dependencies

- **CUDA**: 11.7+ (for GPU support)
- **Python**: 3.10+
- **PyTorch**: 2.0.1+ with CUDA support
- **diamond**: 0.9.14
- **mmseqs2**: 13.4511
- **pyg**: 2.3.1
- **BioPython**: 1.81 (or sequence parsing)
- **fair-esm**: 2.0.0 for esm2
- **scikit-learn**: For similarity calculations
- **NetworkX and obonet**: For ontology processing
- **SciPy**: For sparse matrix operations
- **Pandas**: For data manipulation