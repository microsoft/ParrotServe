#!/bin/sh

# rm *.log -rf
rm model_worker_* -rf
rm tmp/*.txt

bash fastchat/launch_vllm.sh

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

sleep 1

python3 start_benchmark_vllm.py &> 6.log

sleep 1

bash ../../scripts/kill_all_fastchat_servers.sh