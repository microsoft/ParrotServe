#!/bin/sh

# Run vLLM benchmark
rm *.log -rf
bash ../fastchat_scripts/launch_vllm.sh
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
sleep 1
python3 bench_arxiv_multivm_langchain.py > result_vllm.txt
sleep 1
bash ../../scripts/kill_all_fastchat_servers.sh

sleep 3

# Run parrot benchmark
rm -rf log
pwd=$PWD
log_path=$pwd/log/
echo $log_path

cd cluster_1_vicuna_13b_fifo/
bash launch.sh $log_path core.log engine.log
cd ..
python3 bench_arxiv_multivm_parrot.py > result_parrot.txt
sleep 1
bash ../../scripts/kill_all_servers.sh

# Plot the results
python3 plot.py