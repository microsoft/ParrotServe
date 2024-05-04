#!/bin/sh

rm result_parrot.txt
touch result_parrot.txt

counter=1

for r in 1.0 2.0 4.0 6.0 8.0 16.0 32.0
do
    rm -rf log

    pwd=$PWD
    log_path=$pwd/log/

    echo $log_path

    echo "Test GPTs Serving: Parrot (request rate: $r) [$counter / 6]"
    
    if [ $counter -eq 1 ]; then
        num_prompts=300
    else
        num_prompts=500
    fi
    # num_prompts=500
    
    
    counter=$((counter+1))

    # Launch cluster
    cd cluster_4_vicuna_7b_shared
    bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log

    # Run benchmark
    cd ..
    python3 benchmark_serving_parrot.py --workload-info "../workloads/gpts/top4.json" \
        --num-prompts $num_prompts \
        --request-rate $r \
        >> result_parrot.txt

    # Kill cluster
    bash ../../scripts/kill_all_servers.sh
done