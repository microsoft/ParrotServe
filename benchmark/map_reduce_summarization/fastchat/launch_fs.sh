#!/bin/sh

python3 -m fastchat.serve.controller &

sleep 1

python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-13b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --stream-interval 9999 \
    --limit-worker-concurrency 2 \
    --seed 0 &

sleep 20

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

