#!/bin/sh

rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd cluster_1_vicuna_13b_ctx
bash launch.sh $log_path core.log engine.log

# Run benchmark
cd ..
python3 bench_hack_parrot.py > result_parrot.log # > log/program.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_servers.sh