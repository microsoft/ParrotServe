#!/bin/sh

# Run vLLM benchmark

export VLLM_CAPACITY=4096

rm result_vllm.txt
touch result_vllm.txt
for i in {0..9}
do
    rm *.log -rf
    bash ../fastchat_scripts/launch_vllm.sh
    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY
    echo "Run vLLM benchmark ... [$(($i+1)) / 10]"
    python3 bench_arxiv_backgrounds_langchain.py $i >> result_vllm.txt
    bash ../../scripts/kill_all_fastchat_servers.sh
done
