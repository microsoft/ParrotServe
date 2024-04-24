#!/bin/sh

# Run huggingface benchmark
rm *.log -rf
bash ../fastchat_scripts/launch_hf.sh
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
echo "Run HF benchmark ... [All]"
sleep 1
python3 bench_arxiv_langchain.py exp1 -1 > result_hf_olen.txt
sleep 1
python3 bench_arxiv_langchain.py exp2 -1 > result_hf_csize.txt
sleep 1
bash ../../scripts/kill_all_fastchat_servers.sh