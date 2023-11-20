#!/bin/sh

python3 -m fastchat.serve.controller &

sleep 1

python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.3 --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002"  --tokenizer hf-internal-testing/llama-tokenizer &

sleep 10

python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

sleep 1

python3 chain_summarization_langchain_baseline.py

ps -ef | grep fastchat | grep -v grep | awk '{print $2}' | xargs kill -9