#!/bin/sh

VLLM_CAPACITY=$1 python3 -m vllm.entrypoints.api_server \
    --model lmsys/vicuna-13b-v1.3 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --swap-space 16 \
    --disable-log-requests \
    --max-num-batched-tokens 2560 &> $2 &