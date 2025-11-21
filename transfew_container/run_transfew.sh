#!/bin/bash
set -e

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Run Python with all passed arguments
python transfew_main.py "$@"