#!/bin/sh

rm vllm_server.log
rm vllm_client.log

# Launch server
bash start_vllm_server.sh $1 "vllm_server.log"
sleep 60

# Run benchmark
bash start_benchmark_vllm.sh $2 &> "vllm_client.log"
sleep 1

# Kill cluster
bash ../../scripts/kill_all_vllm_servers.sh