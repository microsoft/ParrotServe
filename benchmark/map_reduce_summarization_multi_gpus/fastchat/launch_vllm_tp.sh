#!/bin/sh

python3 -m fastchat.serve.controller &

sleep 1

python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --seed 0 \
    --port 21002 \
    --tensor-parallel-size 4 \
    --num-gpus 4 &

sleep 20

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

