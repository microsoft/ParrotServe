#!/bin/sh

echo "Start OS server ..."
python3 -m parrot.os.http_server --config_path sample_configs/os/localhost_os.json --log_dir log/ --log_filename os_4_opt_125m.log &

sleep 1
for i in {1..4} 
do  
    echo "Start OPT-125m server ($i/4) ..."
    python3 -m parrot.engine.http_server \
        --config_path sample_configs/engine/opt-125m.json \
        --log_dir log/ \
        --log_filename engine_opt_server_$i.log \
        --port 900$i \
        --engine_name opt_125m_server_$i \
        &
    sleep 1
done

sleep 5

echo "Successfully launched Parrot runtime system."