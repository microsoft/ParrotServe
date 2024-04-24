#!/bin/sh

# Run huggingface benchmark
rm *.log -rf
bash ../fastchat_scripts/launch_hf.sh
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
sleep 1
python3 bench_arxiv_langchain.py exp1 -1 > result_hf_olen.txt
sleep 1
python3 bench_arxiv_langchain.py exp2 -1 > result_hf_csize.txt
sleep 1
bash ../../scripts/kill_all_fastchat_servers.sh

# Run vLLM benchmark
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
    python3 bench_arxiv_langchain.py exp1 $i >> result_vllm_olen.txt
    sleep 1
    python3 bench_arxiv_langchain.py exp2 $i >> result_vllm_csize.txt
    sleep 1
    bash ../../scripts/kill_all_fastchat_servers.sh
done

# Run parrot benchmark
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

cd cluster_1_vicuna_13b/
bash launch.sh $log_path core.log engine.log
cd ..
python3 bench_arxiv_parrot.py exp1 > result_parrot_olen.txt
sleep 1
python3 bench_arxiv_parrot.py exp2 > result_parrot_csize.txt
sleep 1
bash ../../scripts/kill_all_servers.sh

# Plot the results
python3 plot.py