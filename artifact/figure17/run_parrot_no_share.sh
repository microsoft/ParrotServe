#!/bin/sh

rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd cluster_1_vicuna_13b_paged
bash launch.sh $log_path os.log engine.log

# Run benchmark
cd ..
python3 bench_hack_parrot.py no_cache > result_parrot_no_share.txt # > log/program.log

# Kill cluster
bash ../../scripts/kill_all_servers.sh