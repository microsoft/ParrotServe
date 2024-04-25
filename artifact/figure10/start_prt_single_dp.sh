#!/bin/sh
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
python3 dump_engine_config.py $1
cd cluster_1_vicuna_13b_vllm/
bash launch.sh $log_path os.log engine.log

# Run benchmark
cd ..
bash start_benchmark_parrot.sh $2 &> log/client.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_servers.sh