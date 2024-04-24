#!/bin/sh

python3 -m fastchat.serve.controller &> fschat_controller_stdout.log &

sleep 1

python3 -m fastchat.serve.vllm_worker \
     --model-path lmsys/vicuna-13b-v1.3 \
     --model-names "gpt-3.5-turbo" \
     --limit-worker-concurrency 999999 \
     --tokenizer hf-internal-testing/llama-tokenizer \
     --max-num-batched-tokens 8000 \
     --seed 0 &> worker_vllm_stdout.log &

sleep 20

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &> fschat_api_server_stdout.log &

