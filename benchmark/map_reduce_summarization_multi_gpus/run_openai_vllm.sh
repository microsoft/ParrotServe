#!/bin/sh

rm -rf log
rm *.log -rf

bash fastchat/launch_vllm.sh

pwd=$PWD
log_path=$pwd/log/

# Launch cluster
cd openai
bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log
sleep 2

# Run benchmark
cd ..

python3 bench_map_reduce_summarization.py > 1.log

sleep 1

bash ../../scripts/kill_all_fastchat_servers.sh
bash ../../scripts/kill_all_servers.sh