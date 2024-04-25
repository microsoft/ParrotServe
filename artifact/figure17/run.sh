#!/bin/sh

# Run vLLM (latency) benchmark
bash run_vllm_lat.sh

# Run vLLM (throughput) benchmark
bash run_vllm_thr.sh

# Run parrot (w/ PagedAttention) benchmark
bash run_parrot_paged.sh

# Run parrot benchmark
bash run_parrot.sh

# Run memory benchmark: no share
bash run_parrot_no_share.sh

# Plot the results
python3 plot.py