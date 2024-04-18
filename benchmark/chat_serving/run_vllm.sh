#!/bin/sh

# Launch server
bash start_vllm_server.sh
sleep 20

# Run benchmark
bash start_benchmark_vllm.sh &> 2.log
sleep 1

# Kill cluster
bash ../../scripts/kill_all_vllm_servers.sh