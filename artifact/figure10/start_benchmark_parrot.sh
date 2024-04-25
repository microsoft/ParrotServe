#!/bin/sh

python3 benchmark_serving.py \
        --backend parrot \
        --num-prompts 100 \
        --tokenizer hf-internal-testing/llama-tokenizer \
        --dataset ../workloads/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate $1 \