#!/bin/sh

rm result.txt
touch result.txt

# install oldscheduler version
pip install -e . --no-deps

for reqs in {5..25..5}
do
    for capacity in {2048..12288..2048}
    do
    echo "Run with capacity: $capacity, reqs: $reqs"
    echo "capacity: $capacity, reqs: $reqs" >> result.txt
    bash run_bench.sh $capacity $reqs
    python3 parse_log.py >> result.txt
    sleep 5
    done
done

pip uninstall parrot_vllm_oldscheduler -y