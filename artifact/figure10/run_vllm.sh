#!/bin/sh

# warn: rm result
rm result.txt
touch result.txt

counter=1

for reqs in {5..25..5}
do
    for capacity in {2048..12288..2048}
    do
    echo "Run with capacity: $capacity, reqs: $reqs [$counter / 30]"
    echo "capacity: $capacity, reqs: $reqs" >> result.txt
    bash start_vllm_single_dp.sh $capacity $reqs
    python3 parse_log_vllm.py >> result.txt
    counter=$((counter+1))
    done
done

# Plot the results
python3 plot.py