#!/bin/sh

# Prepare models for the artifact
# For A100*1, load 13B model
# For A6000*4, load 7B model

# Run it with:
#  bash prepare_models.sh 7b/13b

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Please provide the model size: 7b or 13b"
    exit 1
fi

cd model_loader_scripts
python3 create_engine.py $1