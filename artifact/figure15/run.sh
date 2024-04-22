#/bin/bash

echo 'mode, batch_size, sf_e2e, sf_model, dfg_e2e, dfg_model, total_e2e' > ./result_32.txt
for mode in 'vllm_shared' 'parrot_shared'
do
for max_gen_length in 200 400 600 800
    do
        python3 ../figure14/bench_shared_prompt_e2e.py -b 32 -m $mode -l $max_gen_length --log-path ./result_32.txt
    done
done

echo 'mode, batch_size, sf_e2e, sf_model, dfg_e2e, dfg_model, total_e2e' > ./result_64.txt
for mode in 'vllm_shared' 'parrot_shared'
do
for max_gen_length in 100 200 300 400 480
    do
        python3 ../figure14/bench_shared_prompt_e2e.py -b 64 -m $mode -l $max_gen_length --log-path ./result_64.txt
    done
done
