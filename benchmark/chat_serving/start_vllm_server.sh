#!/bin/sh

python3 -m vllm.entrypoints.api_server \
    --model lmsys/vicuna-13b-v1.3 \
    --swap-space 16 \
    --disable-log-requests