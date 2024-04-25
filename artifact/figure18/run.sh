#!/bin/sh

# Run vLLM (latency) benchmark
bash run_vllm_lat.sh

# Run vLLM (throughput) benchmark
bash run_vllm_thr.sh

# Run parrot benchmark
bash run_prt.sh

# Plot the results
python3 plot.py