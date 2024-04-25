#!/bin/sh

python3 -m fastchat.serve.controller &> fschat_controller_stdout.log &

sleep 1

CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --limit-worker-concurrency 9999 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --max-num-batched-tokens 8000 \
    --worker-address http://localhost:21002 \
    --seed 0 \
    --port 21002 &

sleep 1

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --limit-worker-concurrency 9999 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --max-num-batched-tokens 8000 \
    --worker-address http://localhost:21003 \
    --seed 0 \
    --port 21003 &

sleep 1

CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --limit-worker-concurrency 9999 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --max-num-batched-tokens 8000 \
    --worker-address http://localhost:21004 \
    --seed 0 \
    --port 21004 &

sleep 1

CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --limit-worker-concurrency 9999 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --max-num-batched-tokens 8000 \
    --worker-address http://localhost:21005 \
    --seed 0 \
    --port 21005 &

sleep 30

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &> fschat_api_server_stdout.log &

