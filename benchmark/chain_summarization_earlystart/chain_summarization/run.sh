#!/bin/sh
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd ../experiment_configs/cluster_1_vicuna/
bash launch.sh $log_path os.log engine.log

# Run benchmark
cd ../../chain_summarization/
python3 bench_chain_summarization.py # > log/program.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_servers.sh