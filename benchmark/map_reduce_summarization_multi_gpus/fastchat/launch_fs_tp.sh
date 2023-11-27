#!/bin/sh

python3 -m fastchat.serve.controller &

sleep 1

python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names "gpt-3.5-turbo" \
    --seed 0 \
    --port 21002 \
    --num-gpus 4 \
    --gpus 0,1,2,3 \
    --worker http://localhost:21002 &
sleep 20

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

