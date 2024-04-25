#!/bin/sh

# Run vLLM benchmark
bash run_vllm.sh

# Run parrot (w/ PagedAttention) benchmark
bash run_parrot_paged.sh

# Run parrot benchmark
bash run_parrot.sh

# Plot the results
python3 plot.py