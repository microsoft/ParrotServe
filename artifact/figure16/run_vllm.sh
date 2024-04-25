#!/bin/sh

rm result_vllm.txt
touch result_vllm.txt

for r in 0.25 0.5 1.0 1.25 2.0
do
    rm *.log -rf

    bash ../fastchat_scripts/launch_vllm_multi.sh

    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY

    python3 benchmark_serving_vllm.py --workload-info "../workloads/gpts/top4.json" \
        --num-prompts 500 \
        --request-rate $r \
        >> result_vllm.txt

    bash ../../scripts/kill_all_fastchat_servers.sh
done