#!/bin/sh

rm *.log -rf

bash ../experiment_configs/fastchat_vllm/launch_vllm.sh

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

sleep 1

python3 map_reduce_summarization_langchain_baseline.py

sleep 1

bash ../experiment_configs/fastchat_vllm/kill.sh