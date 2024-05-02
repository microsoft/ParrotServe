#!/bin/sh

rm result_vllm_lat.txt
touch result_vllm_lat.txt

for i in {1..5}
do
    echo "Test Mixed Serving: vLLM (Latency) [$i / 5]"

    # rm *.log -rf
    rm model_worker_* -rf

    export VLLM_CAPACITY=4096
    export VLLM_REQ_TRACK=1
    
    bash ../fastchat_scripts/launch_vllm_multi.sh

    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY

    python3 start_benchmark_vllm.py &> vllm_client.log

    # Parse results
    python3 parse_vllm_time.py >> result_vllm_lat.txt

    bash ../../scripts/kill_all_fastchat_servers.sh
done