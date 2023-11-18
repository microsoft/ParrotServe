#!/bin/sh
rm -rf log
bash launch_sys.sh
python3 bench_async_comm.py # > log/program.log
sleep 1
bash ../../scripts/stop_all_servers.sh