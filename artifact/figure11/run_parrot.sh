#!/bin/sh

# Run parrot benchmark
rm -rf log

pwd=$PWD
log_path=$pwd/log/

echo $log_path

cd cluster_1_vicuna_13b/
bash launch.sh $log_path core.log engine.log
cd ..
echo "Run Parrot benchmark ... [All]"
python3 bench_arxiv_parrot.py exp1 > result_parrot_olen.txt
sleep 1
python3 bench_arxiv_parrot.py exp2 > result_parrot_csize.txt
sleep 1
bash ../../scripts/kill_all_servers.sh

# Plot the results
python3 plot.py