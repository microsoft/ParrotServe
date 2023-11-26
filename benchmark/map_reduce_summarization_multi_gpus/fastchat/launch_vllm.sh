#!/bin/sh

python3 -m fastchat.serve.controller &

sleep 1

python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.3 --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002"  --tokenizer hf-internal-testing/llama-tokenizer &

sleep 15

python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.3 --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002"  --tokenizer hf-internal-testing/llama-tokenizer &

sleep 15

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

