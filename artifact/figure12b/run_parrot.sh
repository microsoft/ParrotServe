#!/bin/sh

# Run parrot benchmark
rm -rf log
pwd=$PWD
log_path=$pwd/log/
echo $log_path

cd cluster_1_vicuna_13b_fifo/
bash launch.sh $log_path os.log engine.log
cd ..
echo "Run Parrot benchmark ... [All]"
python3 bench_arxiv_multivm_parrot.py > result_parrot.txt
bash ../../scripts/kill_all_servers.sh
