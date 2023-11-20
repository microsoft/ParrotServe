#!/bin/sh
rm -rf log
bash launch_sys.sh
python3 bench_chain_summarization.py # > log/program.log
sleep 1
bash ../../scripts/stop_all_servers.sh