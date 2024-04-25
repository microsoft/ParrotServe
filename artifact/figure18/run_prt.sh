#!/bin/sh

rm result_parrot.txt
touch result_parrot.txt

for i in {1..5}
do
    echo "Test Mixed Serving: Parrot [$i / 5]"

    rm -rf log
    rm *.log -rf

    pwd=$PWD
    log_path=$pwd/log/

    echo $log_path

    # Launch cluster
    cd cluster_4_vicuna_7b
    bash launch.sh $log_path os.log engine1.log engine2.log engine3.log engine4.log

    # Run benchmark
    cd ..

    python3 start_benchmark_parrot.py &> $log_path/client.log

    # Parse results
    python3 parse_parrot_time.py >> result_parrot.txt

    # Kill cluster
    bash ../../scripts/kill_all_servers.sh
done