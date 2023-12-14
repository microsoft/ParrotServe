#/bin/bash

echo 'mode, batch_size, sf_e2e, sf_model, dfg_e2e, dfg_model, total_e2e' > ./shared_prompt_exp_2.csv
for mode in 'vllm_diverged' 'vllm_shared' 'parrot_shared'
do
	for batch_size in 8 16 32 64
    do
        python benchmark/microbench/bench_shared_prompt_e2e.py -m $mode -b $batch_size --use-sample
    done
done

echo 'mode, batch_size, sf_e2e, sf_model, dfg_e2e, dfg_model, total_e2e' > ./shared_prompt_exp_1.csv
for mode in 'vllm_shared' 'parrot_shared'
do
	for max_gen_length in 100 200 300 400 480
    do
        python benchmark/microbench/bench_shared_prompt_e2e.py -m $mode -l $max_gen_length
    done
done
