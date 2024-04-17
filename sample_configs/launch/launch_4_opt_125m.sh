#!/bin/sh

echo "Start ServeCore server ..."
python3 -m parrot.serve.http_server --config_path sample_configs/core/localhost_serve_core.json --log_dir log/ --log_filename core_4_opt_125m.log &

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
        --device cuda:$[$i-1] &
    sleep 1
done

sleep 5

echo "Successfully launched Parrot runtime system."