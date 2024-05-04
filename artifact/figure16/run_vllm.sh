#!/bin/sh

rm result_vllm.txt
touch result_vllm.txt

counter=1

for r in 0.25 0.5 1.0 2.0
do
    rm *.log -rf
    
    export VLLM_REQ_TRACK=1
    export VLLM_CAPACITY=30000

    bash ../fastchat_scripts/launch_vllm_multi.sh

    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY

    echo "Test GPTs Serving: vLLM (request rate: $r) [$counter / 5]"

    if [ $counter -eq 1 ]; then
        num_prompts=50
    elif [ $counter -eq 2 ]; then
        num_prompts=100
    elif [ $counter -eq 3 ]; then
        num_prompts=300
    else
        num_prompts=500
    fi

    counter=$(($counter+1))

    python3 benchmark_serving_vllm.py --workload-info "../workloads/gpts/top4.json" \
        --num-prompts $num_prompts \
        --request-rate $r \
        >> result_vllm.txt

    bash ../../scripts/kill_all_fastchat_servers.sh
done
