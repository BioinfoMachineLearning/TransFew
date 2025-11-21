#!/bin/bash
set -e

# usage  if cache_path is not provided, it will be saved in the default path: ~/.cache/huggingface/hub
# bash ./download_model.sh --cache-path /home/user/hf_cache

# 0 check where all models weight will be saved
DEFAULT_CACHE="$PWD/checkpoints"
# Default value
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cache-path)
            DEFAULT_CACHE="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            shift
            ;;
    esac
done

# 1. Make sure download directory exists
mkdir -p "$DEFAULT_CACHE"
echo "✅ Download path set to: $DEFAULT_CACHE"

# 2.0, download transfew model (26 GB)
wget -c -P "$DEFAULT_CACHE" \
    https://calla.rnet.missouri.edu/rnaminer/tfew/TFewDataset
cd $DEFAULT_CACHE
unzip TFewDataset
echo "✅ Transfew Model download completed and saved at $DEFAULT_CACHE."

# 2.1, downloaded file size 29 GB
wget -c -P "$DEFAULT_CACHE" \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt

echo "✅ esm2_t48_15B_UR50D Model download completed and saved at $DEFAULT_CACHE."

# 2.2, downloaded file size 12 k
wget -c -P "$DEFAULT_CACHE" \
    https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t48_15B_UR50D-contact-regression.pt
echo "✅ esm2_t48_15B_UR50D-contact-regression Model download completed and saved at $DEFAULT_CACHE."
