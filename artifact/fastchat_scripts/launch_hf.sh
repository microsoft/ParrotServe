#!/bin/sh

python3 -m fastchat.serve.controller &> fschat_controller_stdout.log &

sleep 1

python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-13b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --stream-interval 9999 \
    --limit-worker-concurrency 999999 \
    --seed 0 &> worker_hf_stdout.log &

sleep 20

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &> fschat_api_server_stdout.log &

