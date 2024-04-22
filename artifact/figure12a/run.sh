#!/bin/sh

# warn: rm results
rm *.txt

rm -rf log
rm *.log -rf

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Run vLLM benchmark
bash ../fastchat_scripts/launch_vllm.sh
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
sleep 1
python3 bench_arxiv_backgrounds_langchain.py > result_vllm.txt
sleep 1
bash ../../scripts/kill_all_fastchat_servers.sh

sleep 3

# Run parrot benchmark
# cd cluster_1_vicuna_13b/
# bash launch.sh $log_path core.log engine.log
# cd ..
# python3 bench_arxiv_backgrounds_parrot.py > result_parrot.txt
# sleep 1
# bash ../../scripts/kill_all_servers.sh