#!/bin/sh

# Run huggingface benchmark
bash run_hf.sh

# Run vLLM benchmark
bash run_vllm.sh

# Run parrot benchmark
bash run_parrot.sh

# Plot the results
python3 plot.py