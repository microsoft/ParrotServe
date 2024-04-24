#!/bin/sh
python3 -m parrot.serve.http_server \
    --config_path core.json \
    --log_dir $1 \
    --release_mode \
    --log_filename $2 &

sleep 1

python3 -m parrot.engine.http_server \
    --config_path engine.json \
    --log_dir $1 \
    --log_filename $3 \
    --release_mode \
    --port 9001 \
    --engine_name engine_server1 \
    --device cuda &
sleep 30