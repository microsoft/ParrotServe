#!/bin/sh

# Run huggingface benchmark
# bash ../fastchat_scripts/launch_fs.sh
# export OPENAI_API_BASE=http://localhost:8000/v1
# export OPENAI_API_KEY=EMPTY
# sleep 1
# python3 bench_arxiv_langchain.py exp1 > result_hf_olen.txt
# sleep 1
# python3 bench_arxiv_langchain.py exp2 > result_hf_csize.txt
# sleep 1
# bash ../../scripts/kill_all_fastchat_servers.sh

sleep 3

# Run vLLM benchmark
bash ../fastchat_scripts/launch_vllm.sh
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
sleep 1
# python3 bench_arxiv_langchain.py exp1 > result_vllm_olen.txt
sleep 1
python3 bench_arxiv_langchain.py exp2 > result_vllm_csize.txt
sleep 1
bash ../../scripts/kill_all_fastchat_servers.sh

sleep 3

# Run parrot benchmark
# rm -rf log
# rm *.log -rf

# pwd=$PWD
# log_path=$pwd/log/

# echo $log_path

# cd cluster_1_vicuna_13b/
# bash launch.sh $log_path core.log engine.log
# cd ..
# python3 bench_arxiv_parrot.py exp1 > result_parrot_olen.txt
# sleep 1
# python3 bench_arxiv_parrot.py exp2 > result_parrot_csize.txt
# sleep 1
# bash ../../scripts/kill_all_servers.sh

# Plot the results
python3 plot.py