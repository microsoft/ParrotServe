#!/bin/sh

rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

# Launch cluster
cd cluster_4_vicuna_7b
bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log

# Run benchmark
cd ..
python3 benchmark_serving.py --workload-info "../workloads/gpts/top4.json" \
    --num-prompts 500 \
    --request-rate 6 \
    > 1.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_servers.sh