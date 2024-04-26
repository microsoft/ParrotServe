#!/bin/sh

rm result_parrot_paged.txt
touch result_parrot_paged.txt

counter=1

for r in 0.5 1.0 2.0 3.0 4.0 6.0
do
    rm -rf log

    pwd=$PWD
    log_path=$pwd/log/

    echo $log_path

    echo "Test GPTs Serving: Parrot w/ PagedAttention (request rate: $r) [$counter / 6]"
    
    if [ $counter -eq 1 ]; then
        num_prompts=100
    elif [ $counter -eq 2 ]; then
        num_prompts=200
    else
        num_prompts=500
    fi
    
    counter=$((counter+1))

    # Launch cluster
    cd cluster_4_vicuna_7b_paged
    bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log

    # Run benchmark
    cd ..
    python3 benchmark_serving_parrot.py --workload-info "../workloads/gpts/top4.json" \
        --num-prompts $num_prompts \
        --request-rate $r \
        >> result_parrot_paged.txt
    sleep 1

    # Kill cluster
    bash ../../scripts/kill_all_servers.sh
done