#!/bin/sh

rm *.log -rf

echo "Test Multi Agents: vLLM (Latency)"

export VLLM_REQ_TRACK=1
export VLLM_CAPACITY=7096
bash ../fastchat_scripts/launch_vllm.sh

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

python3 bench_multi_agents_vllm.py > result_vllm_lat.txt 2> /dev/null

bash ../../scripts/kill_all_fastchat_servers.sh