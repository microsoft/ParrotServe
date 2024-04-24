#!/bin/sh

# Run vLLM benchmark

export VLLM_CAPACITY=4096

rm result_vllm_olen.txt
rm result_vllm_csize.txt
touch result_vllm_olen.txt
touch result_vllm_csize.txt
for i in {0..9}
do
    rm *.log -rf
    bash ../fastchat_scripts/launch_vllm.sh
    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY
    echo "Run vLLM benchmark ... [$(($i+1)) / 10]"
    sleep 1
    python3 bench_arxiv_langchain.py exp1 $i >> result_vllm_olen.txt 2> /dev/null
    sleep 1
    python3 bench_arxiv_langchain.py exp2 $i >> result_vllm_csize.txt 2> /dev/null
    sleep 1
    bash ../../scripts/kill_all_fastchat_servers.sh
done
