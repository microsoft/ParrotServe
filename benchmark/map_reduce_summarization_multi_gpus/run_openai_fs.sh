#!/bin/sh

rm *.log -rf

bash fastchat/launch_fs.sh

# Launch cluster
cd openai
bash launch.sh log os.log engine.log
sleep 2

# Run benchmark
cd ..

python3 bench_map_reduce_summarization.py

sleep 1

bash ../../scripts/kill_all_fastchat_servers.sh
bash ../../scripts/kill_all_servers.sh