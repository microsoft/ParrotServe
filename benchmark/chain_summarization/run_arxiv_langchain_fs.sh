#!/bin/sh

rm *.log -rf

bash fastchat/launch_fs.sh

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

sleep 1

python3 bench_arxiv_langchain.py > 2.log

sleep 1

bash ../../scripts/kill_all_fastchat_servers.sh