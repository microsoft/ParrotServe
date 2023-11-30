#!/bin/sh
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd cluster_1_vicuna_13b/
bash launch.sh $log_path os.log engine.log

# Run benchmark
cd ..
bash start_benchmark.sh &> 1.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_servers.sh