#/bin/bash

log_path=./result.txt
echo 'mode, batch_size, sf_e2e, sf_model, dfg_e2e, dfg_model, total_e2e, request_completion_time' > $log_path
for mode in 'vllm_diverged' 'vllm_shared' 'parrot_shared'
do
	for batch_size in 8 16 32 64
    do
        python3 bench_shared_prompt_e2e_with_requests.py -m $mode -b $batch_size --use-sample --log-path $log_path
    done
done

# Plot the results
python3 plot.py
