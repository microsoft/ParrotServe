#!/bin/sh

rm result.log
touch result.log

# install oldscheduler version
# pip install -e . --no-deps

for reqs in {5..25..5}
do
    for capacity in {2048..12288..2048}
    do
    echo "Run with capacity: $capacity, reqs: $reqs"
    echo "capacity: $capacity, reqs: $reqs" >> result.log
    # bash run_bench.sh $capacity $reqs
    bash run_vllm.sh $capacity $reqs
    # python3 parse_log.py >> result.log
    python3 parse_log_vllm.py >> result.log
    sleep 5
    done
done

# pip uninstall parrot_vllm_oldscheduler -y