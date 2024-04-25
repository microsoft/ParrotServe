#!/bin/sh

# Run vLLM benchmark
bash run_vllm.sh

# Run parrot (w/ PagedAttention) benchmark
bash run_prt_paged.sh

# Run parrot benchmark
bash run_prt.sh

# Plot the results
python3 plot.py