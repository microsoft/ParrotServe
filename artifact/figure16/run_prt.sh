#!/bin/sh

rm result_parrot.txt
touch result_parrot.txt

counter=1

for r in 16.0
do
    rm -rf log

    pwd=$PWD
    log_path=$pwd/log/

    echo $log_path

    echo "Test GPTs Serving: Parrot (request rate: $r) [$counter / 6]"
    
    # if [ $counter -eq 1 ]; then
    #     num_prompts=100
    # elif [ $counter -eq 2 ]; then
    #     num_prompts=200
    # elif [ $counter -eq 3 ]; then
    #     num_prompts=300
    # elif [ $counter -eq 4 ]; then
    #     num_prompts=300
    # elif [ $counter -eq 5 ]; then
    #     num_prompts=2000
    # elif [ $counter -eq 6 ]; then
    #     num_prompts=4000
    # fi
    num_prompts=500
    
    
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