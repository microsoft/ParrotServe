#!/bin/sh
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd ../experiment_configs/cluster_4_vicuna/
bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log

# Run benchmark
cd ../../map_reduce_summarization/
python3 bench_map_reduce_summarization.py # > log/program.log
sleep 1

# Kill cluster
bash ../../util_scripts/kill_all_servers.sh