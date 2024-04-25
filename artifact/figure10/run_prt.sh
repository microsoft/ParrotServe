#!/bin/sh

rm result.txt
touch result.txt

counter=1

for reqs in {5..25..5}
do
    for capacity in {2048..12288..2048}
    do
    echo "Run with capacity: $capacity, reqs: $reqs [$counter / 30]"
    echo "capacity: $capacity, reqs: $reqs" >> result.txt
    bash start_prt_single_dp.sh $capacity $reqs
    python3 parse_log_parrot.py >> result.txt
    counter=$((counter+1))
    done
done
